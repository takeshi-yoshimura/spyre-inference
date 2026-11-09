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

"""Building a model on the host, ready for the move to Spyre."""

from __future__ import annotations

from collections.abc import Generator
from functools import wraps
from typing import cast
from unittest.mock import patch

import torch
import torch.nn as nn
from vllm import envs as vllm_envs
from vllm.config import LoadConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

from spyre_inference.v1.worker.model_load import install_pad_loaders, fix_padded_model


class SkeletonLoader(DefaultModelLoader):
    """Builds and post-processes the model on meta without reading any weights."""

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        attention_init = Attention.__init__

        @wraps(attention_init)
        def host_attention_init(layer, *args, **kwargs):
            # Attention stays on CPU; KV-scale finalization needs real scalar values.
            with torch.device("cpu"):
                attention_init(layer, *args, **kwargs)

        with set_default_torch_dtype(cast(torch.dtype, model_config.dtype)):
            with torch.device("meta"), patch.object(Attention, "__init__", host_attention_init):
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )
            self.load_weights(model, model_config)
            # Avoid upstream's offload context moving host Attention parameters to meta.
            process_weights_after_loading(model, model_config, torch.device("cpu"))
        return model.eval()


def build_model(
    load_config: LoadConfig, vllm_config: VllmConfig, model_config: ModelConfig
) -> nn.Module:
    """Build and load the model on the host, ready for the move to Spyre.

    With ``load_config.device == "meta"`` the model is only built (see
    ``SkeletonLoader``), for ``spyre_native_cache.restore_cache`` to fill.
    """
    model_loader = (
        SkeletonLoader(load_config)
        if load_config.device == "meta"
        else get_model_loader(load_config)
    )

    install_pad_loaders(model_loader, model_config)

    # load_model reads the target device from vllm_config.load_config.
    saved_load_config = vllm_config.load_config
    vllm_config.load_config = load_config
    try:
        model = model_loader.load_model(vllm_config=vllm_config, model_config=model_config)
    finally:
        vllm_config.load_config = saved_load_config

    fix_padded_model(model, model_config)

    # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
    # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
    # an nn.Module, but just the attention implementation.
    Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

    return model


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """vLLM's ``--load-format fastsafetensors`` iterator, loading onto the device the
    model is being built on (the host, for Spyre) instead of assuming CUDA.

    The target is host memory, so GDS never applies (nogds). Every rank needs the
    full tensors before TP slicing, and reading them locally beats broadcasting
    them over the CPU (gloo) group, so each rank reads every file (all_local).
    """
    from fastsafetensors import SingleGroup
    from fastsafetensors.parallel_loader import ParallelLoader
    from vllm.model_executor.model_loader.weight_utils import _natural_sort_key, enable_tqdm

    vllm_config = get_current_vllm_config()
    device = vllm_config.load_config.device or vllm_config.device_config.device
    pg = torch.distributed.group.WORLD if torch.distributed.is_initialized() else SingleGroup()
    loader = ParallelLoader(
        pg=pg,
        hf_weights_files=sorted(hf_weights_files, key=_natural_sort_key),
        queue_size=vllm_envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE,
        use_tqdm_on_load=enable_tqdm(use_tqdm_on_load),
        device=str(device),
        nogds=True,
        all_local=True,
    )
    try:
        yield from loader.iterate_weights()
    finally:
        loader.close()


def install_fastsafetensors_iterator() -> None:
    """Route ``--load-format fastsafetensors`` through ``fastsafetensors_weights_iterator``."""
    from vllm.model_executor.model_loader import default_loader, weight_utils

    default_loader.fastsafetensors_weights_iterator = fastsafetensors_weights_iterator
    weight_utils.fastsafetensors_weights_iterator = fastsafetensors_weights_iterator
