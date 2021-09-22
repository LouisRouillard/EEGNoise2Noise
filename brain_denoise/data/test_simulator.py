import pytest
import torch

from brain_denoise.data.simulator import generate_signal, generate_noise, simulate_data
from brain_denoise.utils import set_torch_seed


@pytest.mark.parametrize(
    "signal_type", ["sin", "fitzhugh"]
)
@pytest.mark.parametrize(
    "noise_type", ["gaussian", "step", "dirac"]
)
def test_seeds(signal_type, noise_type):
    ns, nc, nt = 50, 100, 150
    seed_1, seed_2 = 1, 5

    # Test: signal depends on seed
    signal_1 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type, seed=seed_1)
    signal_2 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type, seed=seed_1)
    assert torch.allclose(signal_1, signal_2), \
        "Same seed does not yield same signal: there is a hidden source of stochasticity."

    signal_1 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type, seed=seed_1)
    signal_2 = generate_signal(ns=ns, nc=nc, nt=nt, signal_type=signal_type, seed=seed_2)
    assert ~torch.allclose(signal_1, signal_2), \
        "Different seeds do not yield different signals: signal is deterministic."

    # Test: noise depends on seed
    noise_1 = generate_noise(ns=ns, nc=nc, nt=nt, noise_type=noise_type, seed=seed_1)
    noise_2 = generate_noise(ns=ns, nc=nc, nt=nt, noise_type=noise_type, seed=seed_1)
    assert torch.allclose(noise_1, noise_2), \
        "Same seed does not yield same noise: there is a hidden source of stochasticity."

    noise_1 = generate_noise(ns=ns, nc=nc, nt=nt, noise_type=noise_type, seed=seed_1)
    noise_2 = generate_noise(ns=ns, nc=nc, nt=nt, noise_type=noise_type, seed=seed_2)
    assert ~torch.allclose(noise_1, noise_2), \
        "Different seeds do not yield different noises: noise is deterministic."

    # Test: data (signal + noise) depends on seed
    data_in, data_out, _ = simulate_data(
        ns=ns, nc=nc, nt=nt, 
        signal_type=signal_type, noise_type=noise_type, seed=seed_1
    )
    data_1 = torch.stack([data_in, data_out])
    data_in, data_out, _ = simulate_data(
        ns=ns, nc=nc, nt=nt, 
        signal_type=signal_type, noise_type=noise_type, seed=seed_1
    )
    data_2 = torch.stack([data_in, data_out])
    assert torch.allclose(data_1, data_2), \
        "Same seed does not yield same data: there is a hidden source of stochasticity."

    data_in, data_out, _ = simulate_data(
        ns=ns, nc=nc, nt=nt, 
        signal_type=signal_type, noise_type=noise_type, seed=seed_1
    )
    data_1 = torch.stack([data_in, data_out])
    out = simulate_data(
        ns=ns, nc=nc, nt=nt, 
        signal_type=signal_type, noise_type=noise_type, seed=seed_2
    )
    data_2 = torch.stack([data_in, data_out])
    assert ~torch.allclose(data_1, data_2), \
        "Different seeds do not yield different data: data is deterministic."
