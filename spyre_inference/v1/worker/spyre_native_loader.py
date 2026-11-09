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

"""Moving a model's tensors to Spyre in bulk.

``Module._apply`` is run with ``_Recorder``, a conversion function equivalent to
``t.to(spyre)`` that only allocates each destination and records it, so custom
``_apply`` overrides still decide what moves and how. The data is filled
afterwards by the fastsafetensors Spyre copier's aligned host-buffer pipeline.

``move_model_to_spyre`` is the direct form; ``spyre_native_cache`` replays the
same records to write converted tensors to a file and to restore them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from fastsafetensors.copier.spyre import TransferPipeline, stage_to_spyre
from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_dense(t: torch.Tensor) -> bool:
    """Non-overlapping and dense, i.e. ``_to_copy`` keeps ``t``'s strides."""
    expected = 1
    for stride, size in sorted((s, n) for s, n in zip(t.stride(), t.shape) if n != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _unstageable_reason(t: torch.Tensor) -> str | None:
    """Why ``t`` must take the normal ``t.to()`` path, or None if it can be staged."""
    if t.device.type not in ("cpu", "meta"):
        return f"on {t.device.type}"
    if t.dtype != torch.float16:
        return str(t.dtype)
    if t.numel() == 0 or t.requires_grad or not _is_dense(t):
        return "empty/grad/non-dense"
    return None


@dataclass
class Record:
    """One call of the ``_apply`` conversion function, in call order."""

    kind: str  # "same": already moved; "defer": dst allocated, data pending; "to": moved now
    src: torch.Tensor
    dst: torch.Tensor
    layout: Any = None  # SpyreTensorLayout requested via place_with_layout


class _Recorder:
    """``Module._apply`` fn equivalent to ``t.to(device)`` that defers the data.

    ``__call__`` and ``place_with_layout`` allocate eligible destinations and
    record them; everything else moves right away through ``t.to()``. ``_apply``
    only stores the returned tensors, so they may be filled later.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: list[Record] = []

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.device.type == self.device.type:
            # A submodule reachable from several parents is visited again after moving.
            dst = t.to(self.device)
            kind = "same"
        elif _unstageable_reason(t) is not None:
            dst = t.to(self.device)
            kind = "to"
        else:
            # Same allocation _to_copy makes for a dense source.
            dst = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=self.device)
            kind = "defer"
        # Hold a detached view, not t: _apply rebinds a Parameter's .data to dst,
        # but the view keeps the CPU storage alive until the data is read.
        self.records.append(Record(kind, t.detach(), dst))
        return dst

    def place_with_layout(self, src: torch.Tensor, layout) -> torch.Tensor:
        """``src.to(device, device_layout=layout)``, deferred like ``__call__``."""
        from torch_spyre._C import spyre_empty_with_layout

        if src.device.type == self.device.type:
            # A shared module (e.g. the cached RoPE) re-placing its moved table.
            dst = src.to(self.device, dtype=src.dtype, device_layout=layout)
            kind = "same"
        elif _unstageable_reason(src) is not None:
            dst = src.to(self.device, dtype=src.dtype, device_layout=layout)
            kind = "to"
        else:
            # Same allocation spyre_to makes for an explicit device_layout.
            dst = spyre_empty_with_layout(
                src.size(), src.stride(), src.dtype, layout, device=self.device
            )
            kind = "defer"
        self.records.append(Record(kind, src.detach(), dst, layout))
        return dst


def fill_records(records: list[Record], device: torch.device) -> TransferPipeline:
    """Fill every deferred destination in ``records`` from its host tensor."""
    pipeline = stage_to_spyre([(r.src, r.dst) for r in records if r.kind == "defer"], device)
    for r in records:
        if r.kind == "to":
            pipeline.count(".to()", r.src.nbytes)
    return pipeline


def move_model_to_spyre(model: nn.Module, device: torch.device) -> None:
    """Drop-in for ``model.to(device)`` that stages fp16 tensors through aligned host buffers."""
    t0 = time.time()
    recorder = _Recorder(device)
    model._apply(recorder)
    pipeline = fill_records(recorder.records, device)
    logger.info(
        "spyre_native: moved model to %s in %.1fs (%s)",
        device,
        time.time() - t0,
        pipeline.summary(),
    )
