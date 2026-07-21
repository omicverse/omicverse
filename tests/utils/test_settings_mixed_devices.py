"""``cpu_gpu_mixed_init(devices=...)`` must let users pin a specific GPU on a
shared multi-GPU server (issue #853).

The device is parsed from int / "1" / "cuda:1" / [1] into a single CUDA index,
pinned via ``torch.cuda.set_device`` (so downstream torch kernels that resolve a
bare ``"cuda"`` through ``current_device`` land on it), and recorded in
``ov.settings.device``. ``devices=None`` keeps the historical auto behaviour.
"""
import pytest

torch = pytest.importorskip("torch")
import omicverse as ov


def _restore(prev_mode, prev_device):
    ov.settings.mode = prev_mode
    ov.settings.device = prev_device


def test_mixed_init_parses_and_pins_device():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    prev = (ov.settings.mode, ov.settings.device)
    try:
        # "cuda:0", 0, and [0] all resolve to the same pinned device.
        for spec in ("cuda:0", "0", 0, [0]):
            ov.settings.cpu_gpu_mixed_init(devices=spec)
            assert ov.settings.mode == "cpu-gpu-mixed"
            assert ov.settings.device == "cuda:0"
            assert torch.cuda.current_device() == 0

        # None keeps the historical auto behaviour (no pin recorded).
        ov.settings.cpu_gpu_mixed_init()
        assert ov.settings.mode == "cpu-gpu-mixed"
        assert ov.settings.device is None
    finally:
        _restore(*prev)


def test_mixed_init_out_of_range_raises():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    prev = (ov.settings.mode, ov.settings.device)
    try:
        with pytest.raises(ValueError, match="only .* CUDA device"):
            ov.settings.cpu_gpu_mixed_init(devices=torch.cuda.device_count() + 5)
    finally:
        _restore(*prev)


def test_mixed_init_unparseable_raises():
    prev = (ov.settings.mode, ov.settings.device)
    try:
        with pytest.raises(ValueError, match="could not parse devices"):
            ov.settings.cpu_gpu_mixed_init(devices="not-a-gpu")
    finally:
        _restore(*prev)
