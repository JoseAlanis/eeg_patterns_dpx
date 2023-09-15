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
