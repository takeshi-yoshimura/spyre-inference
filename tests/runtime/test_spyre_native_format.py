# SPDX-License-Identifier: Apache-2.0
"""Spyre cache payloads and metadata without Spyre hardware."""

import json

import pytest
import torch
from fastsafetensors import ParallelSaver
from fastsafetensors.saver import plain_entry, tensor_fill
from safetensors import safe_open

from spyre_inference.v1.worker.spyre_native_format import (
    KEY_DEVICE,
    KEY_FORMAT,
    KEY_TENSORS,
    KEY_VERSION,
    NativePayload,
    native_entry,
    shard_metadata,
)


@pytest.mark.parametrize("num_shards", [1, 3])
def test_native_payload_and_per_shard_metadata(tmp_path, num_shards):
    src = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    dst = src.t()
    payload = bytes(range(48))
    plain = torch.ones(3)

    def fill(work):
        tensor_fill()([(buf, e) for buf, e in work if not isinstance(e.source, NativePayload)])
        for buf, entry in work:
            if isinstance(entry.source, NativePayload):
                assert entry.source.src is src and entry.source.dst is dst
                buf[:] = payload

    paths = ParallelSaver(
        num_shards=num_shards, fill=fill, shard_metadata=shard_metadata
    ).save_entries(
        [
            native_entry("w", src, dst, nbytes=len(payload), layout={"tiles": 2}),
            plain_entry("plain", plain),
        ],
        str(tmp_path / "native"),
    )
    seen = set()
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            metadata = checkpoint.metadata()
            assert metadata[KEY_VERSION] == "1"
            assert metadata[KEY_DEVICE] == "spyre"
            assert metadata[KEY_FORMAT] == "torch_spyre/1"
            specs = json.loads(metadata[KEY_TENSORS])
            names = set(checkpoint.keys())
            assert set(specs) == names - {"plain"}
            if "w" in names:
                assert specs["w"] == {
                    "dtype": "F16",
                    "shape": [4, 4],
                    "stride": [1, 4],
                    "layout": {"tiles": 2},
                }
                assert checkpoint.get_tensor("w").numpy().tobytes() == payload
            if "plain" in names:
                torch.testing.assert_close(checkpoint.get_tensor("plain"), plain)
            seen.update(names)
    assert seen == {"w", "plain"}


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_native_size(size):
    t = torch.ones(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="must be positive"):
        native_entry("w", t, t, nbytes=size, layout={})


def test_native_entry_size_is_not_a_backend_transfer_limit():
    t = torch.ones(2, dtype=torch.float16)
    nbytes = (4 << 30) + 1
    assert native_entry("w", t, t, nbytes=nbytes, layout={}).nbytes == nbytes
