# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side fastsafetensors loading for Spyre's CPU model construction."""

from collections.abc import Generator

import torch
from vllm import envs as vllm_envs
from vllm.config import get_current_vllm_config


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Read onto the model's build device instead of assuming CUDA.

    Spyre builds the model on the host, so GDS does not apply. Each TP rank needs
    the full tensors before slicing and reads its own files rather than paying
    for broadcasts through the CPU process group.
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
    """Install the host-aware iterator for ``--load-format fastsafetensors``."""
    from vllm.model_executor.model_loader import default_loader, weight_utils

    default_loader.fastsafetensors_weights_iterator = fastsafetensors_weights_iterator
    weight_utils.fastsafetensors_weights_iterator = fastsafetensors_weights_iterator
