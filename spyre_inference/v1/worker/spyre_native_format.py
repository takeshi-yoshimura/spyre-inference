# SPDX-License-Identifier: Apache-2.0
"""Device-native payloads in standard safetensors files.

A device-native tensor is stored as the bytes a device keeps it in (usually
``U8 [nbytes]``), which only that device's backend can interpret. The files stay
standard safetensors; everything else lives in ``__metadata__`` (string values
only):

``fst.native.version``  this convention's version (``NATIVE_VERSION``)
``fst.native.device``   the device backend the payloads are for (e.g. ``spyre``)
``fst.native.format``   the backend's own format id; only the backend judges it
``fst.native.tensors``  JSON ``{name: {"dtype", "shape", "stride", "layout"}}``:
                        the logical tensor each payload holds; ``layout`` is
                        opaque here and handed to the backend untouched

Tensors not listed in ``fst.native.tensors`` are ordinary safetensors tensors.
The format is owned by spyre-inference; fastsafetensors writes opaque bytes
and string metadata without interpreting this convention.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fastsafetensors.saver import WriteEntry
from fastsafetensors.st_types import DType

NATIVE_VERSION = 1

KEY_VERSION = "fst.native.version"
KEY_DEVICE = "fst.native.device"
KEY_FORMAT = "fst.native.format"
KEY_TENSORS = "fst.native.tensors"


@dataclass(frozen=True)
class NativePayload:
    src: Any
    dst: Any
    spec: Mapping[str, Any]


def native_entry(name: str, src: Any, dst: Any, *, nbytes: int, layout: Any) -> WriteEntry:
    from fastsafetensors.frameworks._torch import fst_dtype

    if nbytes <= 0:
        raise ValueError(f"{name}: native payload size must be positive")
    return WriteEntry(
        name,
        DType.U8,
        [nbytes],
        nbytes,
        source=NativePayload(
            src,
            dst,
            {
                "dtype": fst_dtype(dst.dtype).value,
                "shape": list(dst.shape),
                "stride": list(dst.stride()),
                "layout": layout,
            },
        ),
    )


def shard_metadata(entries: Sequence[WriteEntry]) -> dict[str, str]:
    specs = {e.name: e.source.spec for e in entries if isinstance(e.source, NativePayload)}
    return {
        KEY_VERSION: str(NATIVE_VERSION),
        KEY_DEVICE: "spyre",
        KEY_FORMAT: "torch_spyre/1",
        KEY_TENSORS: json.dumps(specs, separators=(",", ":")),
    }
