import torch
import numpy as np
import math


def generate_signal(ns, nc, nt, signal_type="sin"):
    """
    Return torch tensor of shape (ns, nc, nt)
    """

    time = torch.linspace(0, 1, nt)
    if signal_type == "sin":
        freq = 8  # alpha range
        signal = torch.stack([torch.sin(2 * math.pi * freq * time)] * ns, dim=0)
        # Channels
        signal = torch.stack([signal] * nc, dim=1)
    signal -= signal.mean()
    signal /= signal.std()

    return signal


def generate_noise(shape, noise_types=["gaussian"]):
    """
    Return torch tensor of shape
    """
    noise = torch.zeros(shape)

    # Gaussian
    if "gaussian" in noise_types:
        noise += torch.randn(shape)

    # Step
    if "step" in noise_types:
        for i in range(ns):
            step_start = torch.randint(0, nt, (1,))[0]
            step_end = torch.randint(step_start, nt, (1,))[0]
            noise[i, :, step_start:step_end] += 1

    # Dirac
    if "dirac" in noise_types:
        for i in range(ns):
            dirac = torch.randint(0, nt, (1,))[0]
            noise[i, :, dirac] += 100
    return noise


def simulate_data(ns, nc, nt, signal_type="sin", noise_types=["gaussian"], seed=44):
    """
    Return data_in and data_out of shape (ns, nc, nt), two noisy versions
    of the same signal.
    """

    assert signal_type in ["sin"]
    # assert noise_type in ["gaussian", "step", "dirac"]

    signal = generate_signal(ns, nc, nt, signal_type=signal_type)

    # Noise1
    noise_in = generate_noise(signal.shape, noise_types=noise_types)
    data_in = signal + noise_in

    # Noise2
    noise_out = generate_noise(signal.shape, noise_types=noise_types)
    data_out = signal + noise_out

    return data_in, data_out, signal