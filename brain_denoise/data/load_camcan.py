from pathlib import Path
import torch
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath, get_entity_vals
from tqdm import tqdm


def collect_data():
    # %%
    # task = "smt"
    task = "passive"

    bids_root = Path(f"/data/parietal/store/data/camcan/BIDSsep/{task}")
    deriv_root = Path(f"/data/parietal/store/derivatives/camcan/BIDSsep/{task}")

    # Run on one age group
    participants = pd.read_csv(bids_root / "participants.tsv", delimiter="\t")
    participants = participants.query("age < 30")
    participants["subject"] = participants.participant_id.str[4:]

    subjects = participants["subject"].to_list()
    # subjects = sorted(get_entity_vals(deriv_root, entity_key="subject"))
    # # subject = subjects[0]

    # %%
    n_subjects = 10
    n_conditions = 2
    n_epochs = 60
    n_channels = 309
    n_times = 700

    all_epochs = np.zeros((n_subjects, n_conditions, n_epochs, n_channels, n_times))

    for idx_subject, subject in enumerate(subjects[:n_subjects]):
        # for condition in ['audio/1200Hz', 'audio/300Hz', 'audio/300Hz', 'vis/checker']:
        for idx_condition, condition in enumerate(['audio', 'vis/checker']):
            bp = BIDSPath(
                root=deriv_root,
                subject=subject,
                task=task,
                suffix="epo",  # ave
                session=task,
                check=False,
                extension=".fif",
                datatype="meg",
                processing="clean"  # for epochs
            )
            epochs = mne.read_epochs(bp)[condition]
            data = epochs.get_data()[..., :700]   # shape (epoch, channel, time)
            all_epochs[idx_subject][idx_condition] = data

    return all_epochs


def create_samples(all_epochs, n_samples=10000, seed=0):
    """all_epochs is (n_subjects, n_conditions, n_epochs, n_channels, n_times)"""
    n_subjects, n_conditions, n_epochs, n_channels, n_times = all_epochs.shape

    rng = np.random.default_rng(seed)

    data_ins, data_outs = [], []
    for sample in range(n_samples):
        idx_subject = rng.choice(range(n_subjects))
        idx_condition = rng.choice(range(n_conditions))
        idx_epoch_1 = rng.choice(range(n_epochs))

        # sample epoch 2 different to epoch 1
        sel_remove = np.arange(n_epochs) == idx_epoch_1
        pool_epoch_2 = np.arange(n_epochs)[~sel_remove].tolist()
        idx_epoch_2 = rng.choice(pool_epoch_2)
    
        data_in = all_epochs[idx_subject, idx_condition, idx_epoch_1, :]
        data_out = all_epochs[idx_subject, idx_condition, idx_epoch_2, :]

        data_ins.append(data_in)
        data_outs.append(data_out)

    return np.array(data_ins), np.array(data_outs)


# main
all_epochs = collect_data()
# torch.save(all_epochs, "./camcan_data.th")
data_in, data_out = create_samples(all_epochs)
# reformat into torch Tensor
data_in, data_out = torch.Tensor(data_in), torch.Tensor(data_out)
torch.save(data_in, "/storage/store2/work/lchehab/camcan/data_in.th")
torch.save(data_out, "/storage/store2/work/lchehab/camcan/data_out.th")
