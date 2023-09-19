"""
==================================================
Run multivariate pattern analyses for cue activity
==================================================

Fits a classifier to test how well the scalp-topography (i.e., the
multivariate pattern) evoked by the presentation of a stimulus can
be used to discriminate among classes of stimuli across at a given
time point of the EEG epoch.

Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import sys

import os
from pathlib import Path


import numpy as np

from mne.utils import logger

from mvpa import run_gat

# all parameters are defined in config.py
from config import (
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
overwrite = False
jobs = 1
contrast = 'ax_bx'

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        contrast=contrast,
        overwrite=overwrite,
        jobs=jobs
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    contrast = defaults["contrast"]
    overwrite = defaults["overwrite"]
    jobs = defaults["jobs"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(f"{subject} is not a valid subject ID.\n"
                     f"Use one of: {SUBJECT_IDS}")

if not os.path.exists(FPATH_DERIVATIVES):
    raise RuntimeError(
        MISSING_FPATH_BIDS_MSG.format(FPATH_DERIVATIVES)
    )

# %%
# paths and overwrite settings
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
# choose decoder

# It can be one or multiple of:
# "ridge", "log_reg", "svm-lin", or "svm-nonlin"
# Here we use a Ridge regression classifier (i.e., least-squares with
# Tikhonov regularisation)
decoder = "lin-svm"
conditions = ("cue_probe_combination == '%s' | cue_probe_combination == '%s'"
              % tuple(contrast.upper().split('_')))

# %%
# run mvpa with chosen decoder

# compute classification scores for participant
logger.info('\n\nRun GAT for subject: %s\n\n' % subject)
scores, predictions, patterns = run_gat(subject, contrast=conditions,
                                        decoder=decoder, jobs=jobs)

# %%
# save gat results

# create path for gat results
FPATH_GAT = os.path.join(FPATH_DERIVATIVES, 'gat', 'results')

# check if directory exists
if not os.path.exists(FPATH_GAT):
    Path(FPATH_GAT).mkdir(parents=True, exist_ok=True)

# path for scores
FPATH_SCORES = os.path.join(FPATH_GAT,
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_gat_scores_%s_%s.npy'
                            % (f'{subject:03}', decoder, contrast)
                            )

# check if directory exists
if not Path(FPATH_SCORES).exists():
    Path(FPATH_SCORES).parent.mkdir(parents=True, exist_ok=True)
else:
    if not overwrite:
        RuntimeError('%s already exists' % Path(FPATH_SCORES))

# save scores
np.save(FPATH_SCORES, scores)

# path for predictions
FPATH_PREDS = os.path.join(FPATH_GAT,
                           'sub-%s' % f'{subject:03}',
                           'sub-%s_gat_predictions_%s_%s.npy'
                           % (f'{subject:03}', decoder, contrast)
                           )

# save predictions
np.save(FPATH_PREDS, predictions)

# path for patterns
FPATH_PATTERNS = os.path.join(FPATH_GAT,
                              'sub-%s' % f'{subject:03}',
                              'sub-%s_gat_patterns_%s_%s.npy'
                              % (f'{subject:03}', decoder, contrast))

# save scores
np.save(FPATH_PATTERNS, patterns)
