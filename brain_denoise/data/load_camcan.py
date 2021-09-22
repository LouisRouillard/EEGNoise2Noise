from importlib import reload
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath, get_entity_vals
from multiviewica import multiviewica
from mvlearn.decomposition import GroupPCA, MultiviewICA
from tqdm import tqdm

import mvica_utils

reload(mvica_utils)

# %%
OUTPUT_PATH = "/storage/store2/work/athual/outputs/_046_mvica_camcan/individual_pca"

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
all_evokeds = []

for subject in subjects[:50]:
    bp = BIDSPath(
        root=deriv_root,
        subject=subject,
        task=task,
        suffix="ave",
        session=task,
        check=False,
        extension=".fif",
        datatype="meg",
    )

    try:
        evokeds = mne.read_evokeds(bp)
        all_evokeds.append(evokeds)
    except Exception:
        pass


# %%
all_evokeds[0]