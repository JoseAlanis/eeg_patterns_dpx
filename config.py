"""
========================
Study configuration file
========================

Configuration parameters and global values that will be used across scripts.

Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
License: BDS-3
"""
import os
import multiprocessing

from pathlib import Path

import numpy as np

import json

from mne.channels import make_standard_montage

# -----------------------------------------------------------------------------
# check number of available CPUs in system
jobs = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(jobs)

# file paths
with open("./paths.json") as paths:
    paths = json.load(paths)

# -----------------------------------------------------------------------------
# file paths
# path to sourcedata (Biosemi files)
FPATH_SOURCEDATA = Path(paths['sourcedata'])
# path to BIDS compliant directory structure
FPATH_BIDS = Path(paths['bids'])
# path to derivatives
FPATH_DERIVATIVES = Path(os.path.join(FPATH_BIDS, 'derivatives'))

# -----------------------------------------------------------------------------
# file templates
FNAME_SOURCEDATA_TEMPLATE = os.path.join(
    str(FPATH_SOURCEDATA),
    "sub-{subject:03}",
    "{dtype}",
    "sub-{subject:03}_dpx_{dtype}{ext}"
)

# -----------------------------------------------------------------------------
# problematic subjects
NO_DATA_SUBJECTS = {}

# originally, subjects from 1 to 151, but some subjects should be excluded
SUBJECT_IDS = np.array(list(set(np.arange(1, 53)) - set(NO_DATA_SUBJECTS)))

# -----------------------------------------------------------------------------
# default messages
MISSING_FPATH_SOURCEDATA_MSG = (
    "\n    > Could not find the path:\n\n    > {}\n"
    "\n    > Did you define the correct path to the data in `config.py`? "
    "See the `FPATH_SOURCEDATA` variable in `config.py`.\n"
)

MISSING_FPATH_BIDS_MSG = (
    "\n    > Could not find the path:\n\n    > {}\n"
    "\n    > Did you define the correct path to the data in `config.py`? "
    "See the `FPATH_BIDS` variable in `config.py`.\n"
)

# eeg markers
event_id = {"start_record": 127,
            "pause_record": 245,
            "cue_a": 70,
            "cue_b1": 71,
            "cue_b2": 72,
            "cue_b3": 73,
            "cue_b4": 74,
            "cue_b5": 75,
            "probe_x": 76,
            "probe_y1": 77,
            "probe_y2": 78,
            "probe_y3": 79,
            "probe_y4": 80,
            "probe_y5": 81,
            "correct_L": 13,
            "correct_R": 12,
            "incorrect_L": 113,
            "incorrect_R": 112
            }

# create eeg montage
montage = make_standard_montage(kind='standard_1020')

# relevant task events
task_events = {
    'cue_a': 1,
    'cue_b1': 2,
    'cue_b2': 3,
    'cue_b3': 4,
    'cue_b4': 5,
    'cue_b5': 6,
    'probe_x': 7,
    'probe_y1': 8,
    'probe_y2': 9,
    'probe_y3': 10,
    'probe_y4': 11,
    'probe_y5': 12,
    'correct_L': 12,
    'correct_R': 13,
    'incorrect_L': 15,
    'incorrect_R': 16,
    'start_record': 17,
    'pause_record': 18,
}

# %%
# get anonymisation parameters
with open("./anonym.json") as anonym:
    anonym = json.load(anonym)

# %%
# noisy channels per subject
noisy_channels = {
    1: ['F7', 'Cz', 'F1', 'FC5', 'C5'],
    2: ['FT8', 'F8', 'F7', 'C6', 'T7', 'T8', 'P10', 'Fz', 'F2', 'F1', 'AFz'],
    3: ['P7', 'P9'],
    4: ['P1', 'P8', 'PO8', 'FC6', 'AF7', 'TP7', 'P10', 'O2', 'F8', 'C3'],
    5: ['FT7', 'FT8', 'P9'],
    6: ['TP7', 'P9'],
    7: ['AF8', 'Fp2', 'Fp1', 'FT7', 'P10', 'T8', 'TP8'],
    8: [],
    9: ['PO7', 'P9'],
    10: ['Oz'],
    11: [],
    12: ['TP8', 'TP7', 'P9'],
    13: [],
    14: ['CP5', 'PO8'],
    15: ['P9', 'Iz'],
    16: ['T8'],
    17: ['T8'],
    18: ['Fp2', 'P9', 'AF7', 'F1', 'P2', 'TP7'],
    19: ['T8', 'FC6', 'FC5'],
    20: ['P7', 'P8'],
    21: [],
    22: ['AF7', 'Fp1', 'F7'],
    23: [],
    24: ['Fpz', 'AF7', 'PO8', 'CP5', 'FT7', 'T7', 'P9'],
    25: ['FT7'],
    26: [],
    27: ['P5', 'C5', 'F7'],
    28: ['Fp2', 'Fp1', 'Fpz'],
    29: [],
    30: ['F7', 'CP5', 'PO7', 'P1', 'PO8'],
    31: ['AF7', 'Iz', 'FC3'],
    32: [],
    33: ['T8', 'TP8', 'P10', 'CP5', 'FT7'],
    34: [],
    35: ['T7'],
    36: ['AF7', 'Fpz', 'F4', 'FT8'],
    37: ['P9', 'AF7', 'PO7'],
    38: ['AF7'],
    39: [],
    40: ['F7', 'P9', 'T7', 'FT7'],
    41: ['P9', 'T8', 'F6', 'C6', 'P10'],
    42: ['AF4', 'AF8', 'Fz'],
    43: ['AF8', 'FT7', 'F7', 'FC6', 'F4', 'F5', 'F6'],
    44: ['T7', 'TP7'],
    45: ['PO8', 'T8', 'FT8'],
    46: ['Iz', 'Fp2', 'Fpz', 'F8', 'FT8', 'T8', 'F6', 'F4'],
    47: ['T8', 'PO8', 'C6'],
    48: ['AF8', 'C6', 'PO8', 'F3', 'AF3', 'AF4', 'AF7'],
    49: ['FT8', 'C6'],
    50: ['Fpz', 'F5', 'FT7', 'F7', 'PO3'],
    51: ['AF8', 'Iz', 'T8'],
    52: ['FC5', 'F7']
}