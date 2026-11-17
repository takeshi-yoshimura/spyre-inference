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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

pytestmark = pytest.mark.attention


def _impl(dtype, head_size):
    # Isolate allocation from attention compilation and engine configuration.
    impl = SpyreAttentionImpl.__new__(SpyreAttentionImpl)
    impl.staging_rows = 17
    impl.num_heads = 2
    impl.head_size = head_size
    impl.model_dtype = dtype
    impl._staging = None
    impl._group_staging = None
    return impl


@pytest.mark.parametrize("jagged", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [32, 64, 128])
def test_cpu_staging_is_zeroed_and_reused(jagged, dtype, head_size, monkeypatch):
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", str(int(jagged)))
    impl = _impl(dtype, head_size)
    staging = impl._staging_buffers(torch.device("cpu"))
    buffers = [*staging, *([impl._group_staging] if jagged else [])]

    assert (impl._group_staging is not None) == jagged
    for tensor in buffers:
        assert tensor.shape == (17, 2, head_size)
        assert tensor.device.type == "cpu" and tensor.dtype == dtype
        assert torch.count_nonzero(tensor).item() == 0
    assert len({t.data_ptr() for t in buffers}) == len(buffers)
    staging[0].fill_(3)
    assert impl._staging_buffers(torch.device("cpu")) is staging
    assert torch.all(staging[0] == 3)


@pytest.mark.parametrize("jagged", [False, True])
@pytest.mark.parametrize("layout", [None, "row-outermost"])
def test_spyre_staging_allocates_and_zeros_on_device(jagged, layout, monkeypatch):
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", str(int(jagged)))
    impl = _impl(torch.float16, 64)
    device = SimpleNamespace(type="spyre")
    allocations = []
    layout_fn = Mock(return_value=layout)
    monkeypatch.setattr(spyre_attn, "row_outermost_layout", layout_fn)

    def empty(shape, *, dtype, device, device_layout):
        assert shape == (17, 2, 64)
        assert dtype == torch.float16 and device.type == "spyre"
        assert device_layout == layout
        tensor = Mock()
        tensor.zero_.return_value = tensor
        allocations.append(tensor)
        return tensor

    def no_host_staging(*args, **kwargs):
        raise AssertionError("Spyre staging must not allocate or transfer host zeros")

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "zeros", no_host_staging)
    monkeypatch.setattr(spyre_attn, "convert", no_host_staging)
    staging = impl._staging_buffers(device)

    assert len(allocations) == (3 if jagged else 2)
    assert staging == tuple(allocations[:2])
    assert impl._group_staging is (allocations[2] if jagged else None)
    for tensor in allocations:
        tensor.zero_.assert_called_once_with()
    assert impl._staging_buffers(device) is staging
    assert len(allocations) == (3 if jagged else 2)
    layout_fn.assert_called_once_with((17, 2, 64), torch.float16)


@pytest.mark.parametrize("jagged", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [32, 64, 128])
def test_spyre_staging_zero_readback(jagged, dtype, head_size, monkeypatch):
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from torch_spyre._C import get_spyre_tensor_layout

    monkeypatch.setenv("SPYRE_JAGGED_DECODE", str(int(jagged)))
    impl = _impl(dtype, head_size)
    device = torch.device("spyre")
    staging = impl._staging_buffers(device)
    buffers = [*staging, *([impl._group_staging] if jagged else [])]
    expected_layout = spyre_attn.row_outermost_layout((17, 2, head_size), dtype)
    for tensor in buffers:
        assert tensor.device.type == "spyre" and tensor.dtype == dtype
        if expected_layout is not None:
            assert get_spyre_tensor_layout(tensor) == expected_layout
        torch.testing.assert_close(tensor.cpu(), torch.zeros(17, 2, head_size, dtype=dtype))
    assert len({t.data_ptr() for t in buffers}) == len(buffers)
    assert impl._staging_buffers(device) is staging
