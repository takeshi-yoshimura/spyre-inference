# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cache metadata and ordinary tensor restoration, without Spyre hardware."""

import json
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch
from fastsafetensors.copier.spyre import native_shard_metadata
from fastsafetensors.saver import ParallelSaver, plain_entry, save_sharded
from safetensors import safe_open
from safetensors.torch import save_file

from spyre_inference.v1.worker import spyre_native_cache as cache


def test_write_cache_uses_backend_native_writer(tmp_path, monkeypatch):
    from fastsafetensors.copier import spyre
    from fastsafetensors.native import NativeFormat, native_format, native_specs

    from spyre_inference.v1.worker.spyre_native_loader import Record

    runtime = ModuleType("torch_spyre._C")
    calls = []

    def encode(work, num_threads):
        calls.append(num_threads)
        for buf, src, dst in work:
            assert len(buf) == dst.nbytes
            buf[:] = src.reshape(-1).view(torch.uint8).numpy().tobytes()

    runtime.convert_h2d_into = encode
    runtime.dma_bytes = lambda dst: dst.nbytes
    runtime.get_spyre_tensor_layout = lambda dst: {"test_layout": 1}
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = lambda layout: layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)

    class Recorder:
        def __init__(self, device):
            self.records = []

        def __call__(self, src):
            dst = torch.empty_like(src)
            self.records.append(Record("defer", src, dst))
            return dst

    def saver(**kwargs):
        assert kwargs["fill"] is spyre.native_fill
        assert kwargs["shard_metadata"] is spyre.native_shard_metadata
        return ParallelSaver(**kwargs)

    monkeypatch.setattr(cache, "_Recorder", Recorder)
    monkeypatch.setattr(cache, "ParallelSaver", saver)
    model = torch.nn.Module()
    expected = torch.arange(4, dtype=torch.float16)
    model.register_buffer("weight", expected.clone())
    base = str(tmp_path / "cache")
    key = {"model": "cpu-test", "rank": 0}
    assert cache.write_cache(model, torch.device("cpu"), base, key)
    paths = cache.shard_paths(base, cache._NUM_SHARDS)
    framework = cache.get_framework_op("pytorch")
    specs = {}
    for path in paths:
        meta = cache.SafeTensorsMetadata.from_file(path, framework)
        assert native_format(meta) == NativeFormat("spyre", "torch_spyre/1")
        specs.update(native_specs(meta))
        with safe_open(path, framework="pt") as checkpoint:
            names = checkpoint.keys()
            if "t0" in names:
                assert torch.equal(checkpoint.get_tensor("t0"), expected.view(torch.uint8))
    with safe_open(paths[0], framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
    assert metadata[cache._KEY_FORMAT] == cache.FORMAT
    assert json.loads(metadata[cache._KEY_KEY]) == key
    records = json.loads(metadata[cache._KEY_RECORDS])
    assert records[0]["kind"] == "raw" and records[0]["name"] == "t0"
    assert specs["t0"].layout == records[0]["layout"] == {"test_layout": 1}
    assert calls


@dataclass(frozen=True)
class _Padding:
    width: int
    pad: torch.Tensor


class _HostState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.5))
        self.table = {"empty": torch.empty(0)}
        pad = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float16)
        self.vision_buffers = {"segments": _Padding(3, pad)}
        self.nested = [(_Padding(3, pad), {"offsets": torch.arange(3)})]
        self.strided = _Padding(2, torch.arange(6.0).reshape(2, 3).t())
        self.recursive = {"tensor": torch.tensor(7.0)}
        self.recursive["cycle"] = self.recursive

    def _apply(self, fn, recurse=True):
        return self


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(6.0).reshape(2, 3).t())
        self.host = _HostState()


@pytest.fixture
def checkpoint(tmp_path):
    model = _Model()
    entries = [plain_entry("t0", model.weight)]
    records = [
        {
            "kind": "to",
            "name": "t0",
            "placed": None,
            "dense": True,
            **cache._tensor_meta(model.weight),
        }
    ]
    host = []
    for i, (path, _, _, _, t) in enumerate(cache._module_tensors(model.host)):
        name = f"h{i}"
        entries.append(plain_entry(name, t))
        host.append({"path": f"host{path}", "name": name, "dense": True, **cache._tensor_meta(t)})
    base = str(tmp_path / "cache")
    key = {"model": "cpu-test"}
    paths = save_sharded(
        entries,
        base,
        num_shards=cache._NUM_SHARDS,
        shard_metadata=native_shard_metadata,
        metadata={cache._KEY_FORMAT: cache.FORMAT, cache._KEY_KEY: json.dumps(key, sort_keys=True)},
        shard0_metadata={
            cache._KEY_RECORDS: json.dumps(records),
            cache._KEY_HOST: json.dumps(host),
            cache._KEY_ATTRS: json.dumps(cache._attr_signature(model)),
            cache._KEY_FP8: "[]",
        },
    )
    return model, base, key, paths


def test_restore_plain_cache_with_standard_readers(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary CPU restoration must not use Spyre primitives")

    runtime.dma_bytes = unexpected
    runtime.spyre_empty_with_layout = unexpected
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.stride() == model.weight.stride()
    expected = {path: t for path, _, _, _, t in cache._module_tensors(model)}
    for path, _, _, _, t in cache._module_tensors(restored):
        assert t.device.type == "cpu"
        torch.testing.assert_close(t, expected[path])
    assert restored.host.vision_buffers["segments"].pad is restored.host.nested[0][0].pad
    assert restored.host.recursive["cycle"] is restored.host.recursive
    assert restored.host.vision_buffers["segments"].pad.tolist() == [0.0, 0.0, 1.0]


def test_module_tensor_paths_keep_container_keys_distinct():
    model = torch.nn.Module()
    model.table = {1: torch.tensor(1), "1": torch.tensor(2)}
    paths = {path: tensor.item() for path, _, _, _, tensor in cache._module_tensors(model)}
    assert paths == {":item:table[1]": 1, ":item:table['1']": 2}


def test_reject_nested_host_tensor_shape_mismatch(checkpoint, monkeypatch):
    _, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("host shape check must not use DMA")
    runtime.spyre_empty_with_layout = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
        restored.host.vision_buffers["segments"] = _Padding(2, torch.empty(2, dtype=torch.float16))
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.host.vision_buffers["segments"].pad.device.type == "meta"


def test_restore_recorded_host_strides_from_contiguous_meta(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("ordinary host restoration must not use DMA")
    runtime.spyre_empty_with_layout = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
        restored.host.strided = _Padding(2, torch.empty(3, 2))
    assert restored.host.strided.pad.stride() != model.host.strided.pad.stride()
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.host.strided.pad.stride() == model.host.strided.pad.stride()
    torch.testing.assert_close(restored.host.strided.pad, model.host.strided.pad)


@pytest.mark.parametrize("mismatch", ["key", "shard_set"])
def test_reject_cache_before_restoring_tensors(checkpoint, monkeypatch, mismatch):
    _, base, key, paths = checkpoint
    if mismatch == "key":
        key = {"model": "other"}
    else:
        # Repeating a valid shard must fail the existing shard-set validation.
        monkeypatch.setattr(cache, "shard_paths", lambda *args: [paths[0]] * len(paths))
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"
    assert restored.host.scale.device.type == "meta"


def test_fp8_model_uses_normal_transfer(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    monkeypatch.setattr(cache, "_fp8_layers", lambda model: [("linear", model, None)])
    fills = []
    monkeypatch.setattr(cache, "fill_records", lambda records, device: fills.append(records))
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("FP8 cache must not use raw DMA")
    runtime.get_spyre_tensor_layout = runtime.dma_bytes
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    assert not cache.write_cache(model, torch.device("cpu"), base, key)
    assert len(fills) == 1
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"


def test_reject_old_fp8_payload_before_restoring(checkpoint):
    _, base, key, paths = checkpoint
    with safe_open(paths[0], framework="pt") as shard:
        metadata = shard.metadata()
        names = shard.keys()
        tensors = {name: shard.get_tensor(name).clone() for name in names}
    metadata[cache._KEY_FP8] = json.dumps([{"path": "linear", "tiles": []}])
    save_file(tensors, paths[0], metadata=metadata)
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"
