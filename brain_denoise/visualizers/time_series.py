# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

def plot_signals(
    true_signal: np.ndarray,
    noisy_signal: np.ndarray,
    pred_signal: np.ndarray,
) -> None:
    batch_size, n_channels, time_length = true_signal.shape
    fig, axes = plt.subplots(
        ncols=batch_size,
        nrows=n_channels,
        sharex=True,
        figsize=(10 * batch_size, 10 * n_channels)
    )
    fig.suptitle(
        "Time series comparison for Noise2Noise denoising",
        fontsize=30
    )
    ymin = np.min(true_signal)
    ymax = np.max(true_signal)

    t = np.arange(time_length)
    for b in range(batch_size):
        for c in range(n_channels):
            axes[c, b].plot(
                t,
                noisy_signal[b, c],
                "b-",
                alpha=0.1
            )
            axes[c, b].plot(
                t,
                true_signal[b, c],
                "r--"
            )
            axes[c, b].plot(
                t,
                pred_signal[b, c],
                "g-",
            )
            axes[c, b].set_ylim([ymin - 1, ymax + 1])
            axes[c, -1].yaxis.set_label_position('right') 
            axes[c, -1].set_ylabel(f"channel {c}", fontsize=20)
        axes[0, b].xaxis.set_label_position('top') 
        axes[0, b].set_xlabel(f"batch {b}", fontsize=20)

    axes[0, 0].legend(
        [
            "noisy",
            "true",
            "pred"
        ]
    )
