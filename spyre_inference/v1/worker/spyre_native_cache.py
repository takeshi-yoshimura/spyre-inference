# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A loaded model's Spyre tensors, converted, as a Spyre-native checkpoint.

``write_cache`` runs the ``model.to(spyre)`` traversal on a fully loaded host
model and saves what every tensor becomes with ``fastsafetensors.saver``:
native payloads (the device-format bytes a raw DMA sends) where the tensor can
be raw-copied, the host tensor itself where the normal path must move it, and
the tensors that stay on the host after the move. ``restore_cache`` rebuilds
that state on a model built on the meta device: the same traversal allocates
each destination and ``ParallelLoader`` fills them shard by shard, skipping
weight loading, post-processing and conversion.

FP8 models use normal loading; their kernels move weights before this traversal
and are not supported by the host-conversion cache.

Entries are matched to the traversal by call order and to host tensors by
module path, and checked against shape, dtype, strides and layout; the module
tree's plain attributes (ints, strings, shapes, ...) must match too, so state a
meta build derives differently is caught. A mismatch returns False, and the
caller discards the partially restored model and loads normally.

The shards use fastsafetensors' device-native convention in standard safetensors
files. The model-specific replay recipe is JSON in ``__metadata__``; nothing
in the files is executed on load.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.nn as nn
from fastsafetensors import ParallelLoader, SafeTensorsMetadata
from fastsafetensors.copier.spyre import (
    MAX_TRANSFER_BYTES,
    native_entry,
    native_fill,
    native_shard_metadata,
)
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.saver import (
    ParallelSaver,
    plain_entry,
    shard_paths,
    validate_shard_set,
)
from safetensors import safe_open
from vllm.logger import init_logger

from spyre_inference.v1.worker.spyre_native_loader import _is_dense, _Recorder, fill_records

logger = init_logger(__name__)

FORMAT = "spyre-native-cache/3"
_KEY_FORMAT = "spyre_inference.format"
_KEY_KEY = "spyre_inference.key"
_KEY_RECORDS = "spyre_inference.records"
_KEY_HOST = "spyre_inference.host"
_KEY_ATTRS = "spyre_inference.attrs"
_KEY_FP8 = "spyre_inference.fp8"

_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.float8_e4m3fn,
}
_TORCH_DTYPES = {str(d).removeprefix("torch."): d for d in _DTYPES}
# Buffered writes to one file serialize on its inode; separate files write in
# parallel, so the cache is sharded like a checkpoint.
_NUM_SHARDS = 16


class CacheMismatch(Exception):
    """The cache does not describe the model being restored."""


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in _DTYPES:
        raise CacheMismatch(f"dtype {dtype} has no safetensors equivalent")
    return str(dtype).removeprefix("torch.")


def _layout_json(layout) -> dict | None:
    if layout is None:
        return None
    from torch_spyre.layout import layout_to_json

    return layout_to_json(layout)


def _state_tensors(value: Any, path: str, ancestors: frozenset[int] = frozenset()):
    """Tensor leaves in supported host-state containers, without following cycles."""
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, nn.Module) or id(value) in ancestors:
        return
    ancestors = ancestors | {id(value)}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, (str, int)):
                yield from _state_tensors(item, f"{path}[{key!r}]", ancestors)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _state_tensors(item, f"{path}[{index}]", ancestors)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _state_tensors(
                getattr(value, field.name), f"{path}:field:{field.name}", ancestors
            )


def _module_tensors(model: nn.Module) -> Iterator[tuple[str, nn.Module, str, Any, torch.Tensor]]:
    """(path, module, attribute, dict key or None, tensor) for every tensor held."""
    for prefix, module in model.named_modules():
        for name, t in module._parameters.items():
            if t is not None:
                yield f"{prefix}:param:{name}", module, name, None, t
        for name, t in module._buffers.items():
            if t is not None:
                yield f"{prefix}:buffer:{name}", module, name, None, t
        for name, value in vars(module).items():
            if name in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(value, torch.Tensor):
                yield f"{prefix}:attr:{name}", module, name, None, value
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(key, (str, int)):
                        path = f"{prefix}:item:{name}[{key!r}]"
                        for tpath, tensor in _state_tensors(item, path, frozenset({id(value)})):
                            yield tpath, module, name, key, tensor
            elif isinstance(value, (list, tuple)) or is_dataclass(value):
                for tpath, tensor in _state_tensors(value, f"{prefix}:attr:{name}"):
                    yield tpath, module, name, None, tensor


def _simple(value: Any, depth: int = 0) -> str | None:
    """A stable encoding of plain configuration values, None for anything else."""
    if isinstance(value, (bool, int, float, str, type(None), torch.dtype, torch.device)):
        return repr(value)
    if isinstance(value, (tuple, list)) and depth < 2:
        items = [_simple(v, depth + 1) for v in value]
        if all(i is not None for i in items):
            return f"{type(value).__name__}({','.join(items)})"  # type: ignore[arg-type]
    return None


# Attributes a meta build legitimately leaves different and nothing reads after loading.
_IGNORED_ATTRS = {
    # transformers' guard against initializing weights twice.
    "_is_hf_initialized",
}


def _attr_signature(model: nn.Module) -> dict[str, str]:
    sig = {}
    for prefix, module in model.named_modules():
        sig[f"{prefix}:class"] = type(module).__qualname__
        for name, value in vars(module).items():
            if name in _IGNORED_ATTRS:
                continue
            encoded = _simple(value)
            if encoded is not None:
                sig[f"{prefix}:{name}"] = encoded
    return sig


def _signature_diff(expected: dict[str, str], actual: dict[str, str], limit: int = 8) -> str:
    """Differing attributes grouped by name, with a count and one example each."""
    groups: dict[str, list[str]] = {}
    for k in sorted(expected.keys() | actual.keys()):
        if expected.get(k) != actual.get(k):
            path, _, name = k.rpartition(":")
            groups.setdefault(name, []).append(f"{path}: {expected.get(k)} != {actual.get(k)}")
    parts = [f"{name} x{len(v)} (e.g. {v[0]})" for name, v in sorted(groups.items())]
    more = f" (+{len(parts) - limit} more)" if len(parts) > limit else ""
    return "; ".join(parts[:limit]) + more


def _tensor_meta(t: torch.Tensor) -> dict:
    return {"size": list(t.shape), "stride": list(t.stride()), "dtype": _dtype_name(t.dtype)}


def cache_exists(base: str) -> bool:
    return all(os.path.exists(p) for p in shard_paths(base, _NUM_SHARDS))


def _fp8_layers(model: nn.Module):
    for path, layer in model.named_modules():
        owner = getattr(layer, "scheme", None) or getattr(layer, "quant_method", None)
        kernel = getattr(owner, "fp8_linear", None)
        if kernel is None:
            continue
        from spyre_inference.custom_ops.fp8_linear_kernel import SpyreFp8LinearKernel

        if isinstance(kernel, SpyreFp8LinearKernel):
            yield path, layer, kernel


def write_cache(model: nn.Module, device: torch.device, base: str, key: dict) -> bool:
    """Write ``model``'s tensors, as ``model.to(device)`` leaves them, to ``base``'s shards.

    Returns True if the cache was written: device tensors may be allocated but
    not filled, so the model must be discarded (restore from the cache instead).
    Returns False if it could not be written (e.g. no disk space or an
    unsupported tensor); the model has then been moved to ``device`` fully, as
    by ``move_model_to_spyre``.
    """
    from torch_spyre._C import dma_bytes, get_spyre_tensor_layout
    from torch_spyre.layout import layout_to_json

    t0 = time.time()
    recorder = _Recorder(device)
    model._apply(recorder)

    entries = []
    try:
        if any(_fp8_layers(model)):
            raise CacheMismatch("FP8 native caching is not supported by this kernel")
        records = []
        for seq, r in enumerate(recorder.records):
            meta = {"kind": r.kind, **_tensor_meta(r.src), "placed": _layout_json(r.layout)}
            nbytes = dma_bytes(r.dst) if r.kind == "defer" else 0
            if r.kind == "defer" and 0 < nbytes <= MAX_TRANSFER_BYTES:
                layout = layout_to_json(get_spyre_tensor_layout(r.dst))
                meta.update(kind="raw", name=f"t{seq}", layout=layout)
                entries.append(native_entry(f"t{seq}", r.src, r.dst, nbytes=nbytes, layout=layout))
            elif r.kind != "same":
                # The normal path moves it again at restore, from the same host tensor.
                meta.update(kind="to", name=f"t{seq}", dense=_is_dense(r.src))
                entries.append(plain_entry(f"t{seq}", r.src))
            records.append(meta)

        host = []
        seen: set[int] = set()
        for i, (tpath, _, _, _, t) in enumerate(_module_tensors(model)):
            if t.device.type == "meta":
                raise CacheMismatch(f"{tpath} is still on meta after loading")
            if t.device.type != "cpu" or id(t) in seen:
                continue
            seen.add(id(t))
            host.append({"path": tpath, "name": f"h{i}", **_tensor_meta(t), "dense": _is_dense(t)})
            entries.append(plain_entry(f"h{i}", t.detach()))
    except CacheMismatch as e:
        logger.warning("spyre_native: not caching this model (%s); moving it directly.", e)
        fill_records(recorder.records, device)
        return False

    t1 = time.time()
    try:
        saver = ParallelSaver(
            num_shards=_NUM_SHARDS,
            shard_metadata=native_shard_metadata,
            fill=native_fill,
            min_free_bytes=1 << 30,
        )
        paths = saver.save_entries(
            entries,
            base,
            metadata={_KEY_FORMAT: FORMAT, _KEY_KEY: json.dumps(key, sort_keys=True)},
            shard0_metadata={
                _KEY_RECORDS: json.dumps(records),
                _KEY_HOST: json.dumps(host),
                _KEY_ATTRS: json.dumps(_attr_signature(model), sort_keys=True),
                _KEY_FP8: "[]",
            },
        )
    except OSError as e:
        logger.warning("spyre_native: cannot write the cache (%s); moving the model directly.", e)
        fill_records(recorder.records, device)
        return False
    logger.info(
        "spyre_native: wrote %.2f GiB cache %s-*.safetensors in %.1fs (write %.1fs).",
        sum(os.path.getsize(p) for p in paths) / 2**30,
        base,
        time.time() - t0,
        time.time() - t1,
    )
    return True


def _as_recorded(t: torch.Tensor, rec: dict) -> torch.Tensor:
    """A host tensor read back contiguous, given the strides it was recorded with."""
    if rec["dense"] and list(t.stride()) != rec["stride"]:
        out = torch.empty_strided(rec["size"], rec["stride"], dtype=t.dtype)
        out.copy_(t)
        return out
    return t


class _Restorer:
    """``Module._apply`` fn replaying a cache's records in call order.

    Allocates every raw destination (to be filled by ``ParallelLoader``) and
    moves the rest through the normal path right away.
    """

    def __init__(
        self, device: torch.device, headers: dict[str, SafeTensorsMetadata], records: list[dict]
    ) -> None:
        self.device = device
        self.headers = headers
        self.records = records
        self.seq = 0
        self.destinations: dict[str, torch.Tensor] = {}

    def _next(self, t: torch.Tensor, placed) -> dict:
        if self.seq >= len(self.records):
            raise CacheMismatch("more tensors to move than the cache recorded")
        rec = self.records[self.seq]
        if (
            list(t.shape) != rec["size"]
            or str(t.dtype).removeprefix("torch.") != rec["dtype"]
            or _layout_json(placed) != rec["placed"]
        ):
            raise CacheMismatch(
                f"move #{self.seq}: {list(t.shape)} {t.dtype} does not match {rec['size']} "
                f"{rec['dtype']}"
            )
        self.seq += 1
        return rec

    def _restore(self, t: torch.Tensor, placed=None) -> torch.Tensor:
        from torch_spyre._C import dma_bytes, spyre_empty_with_layout

        rec = self._next(t, placed)
        if rec["kind"] == "same":
            if t.device.type != self.device.type:
                raise CacheMismatch(f"move #{self.seq - 1} was a revisit, got {t.device}")
            return self._move(t, placed)
        if rec["kind"] == "to":
            with safe_open(self.headers[rec["name"]].src, framework="pt", device="cpu") as f:
                host = f.get_tensor(rec["name"])
            return self._move(_as_recorded(host, rec), placed)
        if list(t.stride()) != rec["stride"]:
            raise CacheMismatch(
                f"move #{self.seq - 1}: strides {list(t.stride())} != {rec['stride']}"
            )
        from torch_spyre.layout import layout_from_json

        dst = spyre_empty_with_layout(
            rec["size"],
            rec["stride"],
            _TORCH_DTYPES[rec["dtype"]],
            layout_from_json(rec["layout"]),
            device=self.device,
        )
        frame = self.headers[rec["name"]].tensors[rec["name"]]
        if dma_bytes(dst) != frame.data_offsets[1] - frame.data_offsets[0]:
            raise CacheMismatch(f"move #{self.seq - 1}: device size differs from the cache")
        self.destinations[rec["name"]] = dst
        return dst

    def _move(self, src: torch.Tensor, placed) -> torch.Tensor:
        if placed is None:
            return src.to(self.device)
        return src.to(self.device, dtype=src.dtype, device_layout=placed)  # ty: ignore[no-matching-overload]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self._restore(t)

    def place_with_layout(self, src: torch.Tensor, layout) -> torch.Tensor:
        return self._restore(src, layout)


def _swap_in(target: torch.Tensor, data: torch.Tensor) -> None:
    """Give ``target`` ``data``'s contents, keeping its Python object, class and attributes."""
    replacement = torch.Tensor._make_subclass(type(target), data, target.requires_grad)
    replacement.__dict__.update(target.__dict__)
    torch.utils.swap_tensors(target, replacement)


def restore_cache(model: nn.Module, device: torch.device, base: str, key: dict) -> bool:
    """Restore ``model`` (built on meta, not loaded) from ``write_cache``'s shards.

    Returns False if the cache does not match ``key`` or the model; the model
    may be partially restored and must then be discarded.
    """
    t0 = time.time()
    paths = shard_paths(base, _NUM_SHARDS)
    try:
        framework = get_framework_op("pytorch")
        metas = [SafeTensorsMetadata.from_file(p, framework) for p in paths]
        validate_shard_set(metas)
        md = metas[0].metadata if isinstance(metas[0].metadata, dict) else {}
        if md.get(_KEY_FORMAT) != FORMAT or md.get(_KEY_KEY) != json.dumps(key, sort_keys=True):
            raise CacheMismatch("it is for another model, configuration or version")
        records = json.loads(md[_KEY_RECORDS])
        host = json.loads(md[_KEY_HOST])
        expected = json.loads(md[_KEY_ATTRS])
        fp8 = json.loads(md[_KEY_FP8])
        if fp8 or any(_fp8_layers(model)):
            raise CacheMismatch("FP8 native caching is not supported by this kernel")
    except (OSError, ValueError, KeyError, CacheMismatch) as e:
        logger.warning("spyre_native: cache %s unusable (%s).", base, e)
        return False

    headers = {name: meta for meta in metas for name in meta.tensors}
    restorer = _Restorer(device, headers, records)
    try:
        model._apply(restorer)
        if restorer.seq != len(records):
            raise CacheMismatch(f"moved {restorer.seq} tensors, the cache has {len(records)}")
        held = {tpath: t for tpath, _, _, _, t in _module_tensors(model)}
        missing = [rec["path"] for rec in host if rec["path"] not in held]
        if missing:
            raise CacheMismatch(f"{missing[0]} is not in the model")
        for rec in host:
            target = held[rec["path"]]
            if list(target.shape) != rec["size"] or _dtype_name(target.dtype) != rec["dtype"]:
                raise CacheMismatch(f"host tensor {rec['path']} differs from the cache")
            with safe_open(headers[rec["name"]].src, framework="pt", device="cpu") as f:
                data = f.get_tensor(rec["name"])
            _swap_in(target, _as_recorded(data, rec))
        left = [p for p, _, _, _, t in _module_tensors(model) if t.device.type == "meta"]
        if left:
            raise CacheMismatch(f"{len(left)} tensors not in the cache, e.g. {left[0]}")
        actual = _attr_signature(model)
        if expected != actual:
            raise CacheMismatch(f"attributes differ: {_signature_diff(expected, actual)}")
    except (CacheMismatch, KeyError, ValueError, TypeError) as e:
        logger.warning("spyre_native: cache %s does not match the model (%s).", base, e)
        return False
    t1 = time.time()
    loader = ParallelLoader(
        None,
        paths,
        device=str(device),
        destinations=restorer.destinations,
        tensor_filter=restorer.destinations.__contains__,
        all_local=True,
        set_numa=False,
        use_tqdm_on_load=False,
        accumulate_resident=False,
    )
    try:
        for _name, _tensor in loader.iterate_weights():
            pass
    finally:
        loader.close()
    logger.info(
        "spyre_native: restored model from %s in %.1fs (prepare %.1fs, %d tensors loaded).",
        base,
        time.time() - t0,
        t1 - t0,
        len(restorer.destinations),
    )
    return True
