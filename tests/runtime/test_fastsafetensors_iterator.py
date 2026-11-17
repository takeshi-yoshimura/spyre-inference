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

"""``--load-format fastsafetensors`` loads onto the model's build device, not CUDA."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("fastsafetensors")


@pytest.mark.parametrize("configured_device", [None, "cpu"])
@pytest.mark.parametrize("termination", ["exhausted", "closed", "error"])
def test_iterator_configuration_and_cleanup(configured_device, termination, monkeypatch):
    import fastsafetensors.parallel_loader as parallel_loader

    from spyre_inference.v1.worker import fastsafetensors_loader

    config = SimpleNamespace(
        load_config=SimpleNamespace(device=configured_device),
        device_config=SimpleNamespace(device="cuda" if configured_device else "cpu"),
    )
    monkeypatch.setattr(fastsafetensors_loader, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    value = torch.arange(3)

    def weights():
        yield "weight", value
        if termination == "error":
            raise RuntimeError("read failed")

    loader = Mock()
    loader.iterate_weights.side_effect = weights
    factory = Mock(return_value=loader)
    monkeypatch.setattr(parallel_loader, "ParallelLoader", factory)
    iterator = fastsafetensors_loader.fastsafetensors_weights_iterator(
        ["model-10.safetensors", "model-2.safetensors"], use_tqdm_on_load=False
    )
    name, tensor = next(iterator)
    assert name == "weight" and tensor is value
    if termination == "closed":
        iterator.close()
    elif termination == "error":
        with pytest.raises(RuntimeError, match="read failed"):
            next(iterator)
    else:
        assert list(iterator) == []
    kwargs = factory.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert kwargs["hf_weights_files"] == ["model-2.safetensors", "model-10.safetensors"]
    assert kwargs["nogds"] is True and kwargs["all_local"] is True
    assert kwargs["queue_size"] == fastsafetensors_loader.vllm_envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE
    loader.close.assert_called_once_with()


def test_iterator_loads_onto_the_build_device(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from spyre_inference.v1.worker import fastsafetensors_loader as model_build

    tensors = {
        "a.weight": torch.randn(64, 32).half(),
        "b.bias": torch.arange(7, dtype=torch.int64),
    }
    files = []
    for i, (name, t) in enumerate(tensors.items()):
        files.append(str(tmp_path / f"model-{i}.safetensors"))
        save_file({name: t}, files[-1])
    config = SimpleNamespace(
        load_config=SimpleNamespace(device=None),
        device_config=SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(model_build, "get_current_vllm_config", lambda: config)

    got = dict(model_build.fastsafetensors_weights_iterator(files, use_tqdm_on_load=False))

    assert got.keys() == tensors.keys()
    for name, t in got.items():
        assert t.device.type == "cpu", name
        assert torch.equal(t, tensors[name]), name


def _load_on_rank(rank, init_file, files, expected):
    import torch.distributed as dist

    from spyre_inference.v1.worker import fastsafetensors_loader as model_build

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:

        def no_broadcast(*args, **kwargs):
            raise AssertionError("weights must be read locally, not broadcast")

        dist.broadcast = no_broadcast
        config = SimpleNamespace(
            load_config=SimpleNamespace(device=None),
            device_config=SimpleNamespace(device="cpu"),
        )
        model_build.get_current_vllm_config = lambda: config
        got = dict(model_build.fastsafetensors_weights_iterator(files, use_tqdm_on_load=False))
        assert got.keys() == expected.keys()
        for name, t in got.items():
            assert torch.equal(t, expected[name]), name
    finally:
        dist.destroy_process_group()


def test_iterator_reads_every_file_on_each_rank(tmp_path):
    from safetensors.torch import save_file

    tensors = {f"layer{i}.weight": torch.randn(16, 8).half() for i in range(4)}
    files = []
    for i, (name, t) in enumerate(tensors.items()):
        files.append(str(tmp_path / f"model-{i}.safetensors"))
        save_file({name: t}, files[-1])

    torch.multiprocessing.spawn(
        _load_on_rank, args=(str(tmp_path / "init"), files, tensors), nprocs=2
    )


def test_plugin_installs_the_iterator():
    from vllm.model_executor.model_loader import default_loader

    from spyre_inference import register_ops
    from spyre_inference.v1.worker import fastsafetensors_loader as model_build

    register_ops()
    assert (
        default_loader.fastsafetensors_weights_iterator
        is model_build.fastsafetensors_weights_iterator
    )
