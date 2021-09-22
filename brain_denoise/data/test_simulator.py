import pytest
import torch

from brain_denoise.data.simulator import generate_signal, generate_noise, simulate_data

@pytest.mark.parametrize(
    "signal_type", ["sin", "fitzhugh"]
)
def test_seeds(signal_type):
    ns, nc, nt = 100, 300, 200
    seed_1, seed_2 = 1, 5

    # Test: same signal is deterministic
    signal_1 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type)
    signal_2 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type)
    assert torch.allclose(signal_1, signal_2), "Signal is not deterministic."

