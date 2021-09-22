import torch
import numpy as np
import math

from brain_denoise.utils import set_torch_seed


def generate_fitzhugh_nagumo(v_init=0.1, w_init=0.0, forcing=np.zeros(100)):
    """
    Generates a biological neuron model via
    Euler-discretized Fitzhugh-Nagumo dynamics.

    Notations:
    I is the external forcing (electrical current).
    V is the membrane voltage.
    W is the adaptation.
    """
    n_times = len(forcing)

    # harcoded
    a = -0.3
    b = 1.4
    tau = 20
    R = 1

    # Instantiate trajectories
    vs, ws = [], []

    # Initial state
    v, w = v_init, w_init
    vs.append(v)
    ws.append(w)

    # Generate trajectory
    for t in range(n_times - 1):  # init counts as one step
        # Voltage (v) update
        v = 2 * v - (1.0 / 3) * v ** 3 - w + R * forcing[t]
        vs.append(v)

        # Adaptation (w) update
        w = w + (1.0 / tau) * (v + a - b * w)
        ws.append(w)

    potential = torch.Tensor(vs)

    return potential


def generate_signal(ns, nc, nt, signal_type="sin"):
    """
    Return torch tensor of shape (ns, nc, nt)
    """
    time = torch.linspace(0, 1, nt)
    if signal_type == "sin":
        freq = 8  # alpha range
        signal = torch.sin(2 * math.pi * freq * time)
    elif signal_type == "fitzhugh":
        signal = generate_fitzhugh_nagumo(forcing=np.zeros(nt))

    # duplicate signal over samples
    signal = torch.stack([signal] * ns, dim=0)

    # duplicate signal over channels
    signal = torch.stack([signal] * nc, dim=1)

    # normalize the signal
    signal -= signal.mean()
    signal /= signal.std()

    return signal


def generate_noise(ns, nc, nt, noise_type=["gaussian"], seed=44):
    """
    Return torch tensor of shape ns, nc, nt
    """
    set_torch_seed(seed)

    # Instantiate noise
    noise = torch.zeros((ns, nc, nt))
    
    # Gaussian
    if "gaussian" in noise_type:
        noise += torch.randn((ns, nc, nt))

    # Step
    if "step" in noise_type:
        for idx in range(ns):
            step_start = torch.randint(0, nt, (1,))[0]
            step_end = torch.randint(step_start, nt, (1,))[0]
            noise[idx, :, step_start:step_end] += 1

    # Dirac
    if "dirac" in noise_type:
        for i in range(ns):
            dirac = torch.randint(0, nt, (1,))[0]
            noise[i, :, dirac] += 100
    return noise


def simulate_data(ns, nc, nt, signal_type="sin", noise_type=["gaussian"], seed=44):
    """
    Return data_in and data_out of shape (ns, nc, nt), two noisy versions
    of the same signal.
    """

    assert signal_type in ["sin", "fitzhugh"]
    assert noise_type in ["gaussian", "step", "dirac"]

    signal = generate_signal(ns, nc, nt, signal_type=signal_type)

    # Noise1
    noise_in = generate_noise(ns, nc, nt, noise_type=noise_type, seed=seed)
    data_in = signal + noise_in

    # Noise2
    noise_out = generate_noise(ns, nc, nt, noise_type=noise_type, seed=seed+1)
    data_out = signal + noise_out

    return data_in, data_out, signal