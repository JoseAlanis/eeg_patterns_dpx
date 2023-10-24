"""
=============================================
Plot results of multivariate pattern analysis
=============================================
Creates figures to show classifier performance at multiple time points of
the EEG epoch.

Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import sys

import glob

import os

import numpy as np

from mvpa import get_p_scores, get_stats_lines
from stats import ModelInference, bootstrap_ttest

from viz import plot_gat_matrix

from mne import read_epochs, EvokedArray
from mne.decoding import Vectorizer
from mne.utils import logger

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
report = False
jobs = 1

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=subject,
        overwrite=overwrite,
        report=report,
        jobs=jobs
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    overwrite = defaults["overwrite"]
    report = defaults["report"]

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

# choose decoder
decoder = 'lin-svm'
# condition
cue = True
contrast = ('_ax_bx' if cue else '_ax_ay')

# create path for GAT scores
FPATH_GAT = os.path.join(FPATH_DERIVATIVES, 'gat')
FPATH_GAT_SCORES = os.path.join(FPATH_GAT,
                                'results',
                                'sub-*',
                                'sub-*_gat_%s_%s%s.npy'
                                % ('scores', decoder, contrast))
FPATH_GAT_SCORES = glob.glob(FPATH_GAT_SCORES)
FPATH_GAT_SCORES.sort()

# create path for GAT patterns
FPATH_GAT_PATTERNS = os.path.join(FPATH_GAT,
                                  'results',
                                  'sub-*',
                                  'sub-*_gat_%s_%s%s.npy'
                                  % ('patterns', decoder, contrast))
FPATH_GAT_PATTERNS = glob.glob(FPATH_GAT_PATTERNS)
FPATH_GAT_PATTERNS.sort()

# %%
# create path for preprocessed dara
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'gat/epochs',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_cues-epo.fif' % f'{subject:03}')
# get the data
cue_epo = read_epochs(FPATH_EPOCHS, preload=False)
epochs_info = cue_epo.info
tmin = cue_epo.tmin
times = cue_epo.times
n_times = times.shape[0]
channels = cue_epo.ch_names
n_channels = len(channels)
# shape of eeg data (channels x times)
eeg_shape = (n_channels, n_times)

# %%
# load subject level results

# classifier scores
shape_like = np.load(FPATH_GAT_SCORES[0]).shape
scores = np.empty((len(FPATH_GAT_SCORES), shape_like[0], shape_like[1]))
for nsubj, fpath in enumerate(FPATH_GAT_SCORES):
    print(fpath)
    scores[nsubj, ...] = np.load(fpath)

# classifier patterns
patterns = np.empty((len(FPATH_GAT_PATTERNS), eeg_shape[0], eeg_shape[1]))
for nsubj, fpath in enumerate(FPATH_GAT_PATTERNS):
    patterns[nsubj, ...] = np.load(fpath).reshape(eeg_shape)

# mean of subjects
mean_patterns = patterns.mean(axis=0)

# %%
# run bootstrap test on classification scores to assess significance
bootstrap = True
method = ('bootstrap' if bootstrap else 'permutations')
threshold = 'fmax'
fname_fmax = os.path.join(
    FPATH_DERIVATIVES,
    'gat',
    'gat%s_%s_%s.npy' % (contrast, method, threshold)
)

zero = np.where(times == -.25)[0][0]
scores_flat = scores[:, zero:, zero:]
scores_flat = Vectorizer().fit_transform(scores_flat)

if not os.path.isfile(fname_fmax):
    if bootstrap:
        # compute p values via bootstrap
        chance = np.empty(scores_flat.shape)
        chance[:] = 0.0
        f_max = bootstrap_ttest(data_one=scores_flat - 0.5, data_two=chance,
                                one_sample=True,
                                nboot=2000,
                                multcomp=threshold,
                                random=True,
                                jobs=jobs)
        np.save(fname_fmax, f_max)

        # get observed t-values
        MI = ModelInference()
        MI.paired_ttest(data_one=scores_flat - 0.5, data_two=chance)
        # extract test-values
        t_vals = MI.t_vals_.copy()
        # get sig thresholds
        sig_mask = t_vals ** 2 > np.quantile(f_max, [.99], axis=0)
        sig_mask = sig_mask.reshape(scores[:, zero:, zero:].mean(axis=0).shape)
else:
    if bootstrap:
        sig_mask = np.load(fname_fmax)

# %%
# create generalisation across time (GAT) matrix figure

# classifier performance for specific time slices of interest
if cue:
    test_times = {'[1]   180 - 190  ms': [0.18, 0.19],
                  '[2]   550 - 650  ms': [0.53, 0.63],
                  '[3] 1000 - 1100 ms': [1.0, 1.1]}
else:
    test_times = {'[1] 2700 - 2800 ms': [2.7, 2.8],
                  '[2] 2850 - 2950 ms': [2.85, 2.95],
                  '[3] 3000 - 3100 ms': [3.00, 3.10]}

# compute significance for those time slices
stats_dict = get_stats_lines(scores[:, zero:, zero:],
                             times=times[times >= -0.25],
                             test_times=test_times,
                             correction='Bonferroni')

# plot GAT performance
fig = plot_gat_matrix(data=scores[:, zero:, zero:].mean(0),
                      times=times[times >= -0.25],
                      stats_dict=stats_dict,
                      mask=(sig_mask if bootstrap else None),
                      vmax=0.7, vmin=0.3,
                      draw_mask=False, draw_contour=False,
                      draw_diag=True, draw_zerolines=True,
                      xlabel="Time (ms)", ylabel="Time (ms)",
                      title_gat=('GAT (AX vs. BX)' if cue else 'GAT (AX vs. AY)'),
                      cmap="RdBu_r", mask_cmap="RdBu_r", mask_alpha=1.0,
                      test_times=test_times,
                      focus=None,
                      legend_loc='upper center',
                      figsize=(9.0, 6.0))
gat_figure_path = os.path.join(
    FPATH_DERIVATIVES,
    'gat',
    'gat_matrix_%s_%s%s.tiff'
    % (method, threshold, contrast)
)
fig.savefig(gat_figure_path, dpi=600)

# %%

# put patterns in mne.Evoked format
classifier_patterns = EvokedArray(mean_patterns, cue_epo.info, tmin)
classifier_patterns.plot_joint(times=[0.17, 2.45])

# %%
# supplemental analysis ax vs bx probe
probe_test_times = {'Probe N170/P2': [2.7, 2.8],
                    'Probe P3a': [2.85, 2.95],
                    'Probe CPP': [3.00, 3.10]}
# compute significance for those time slices
stats_dict = get_stats_lines(scores[:, zero:, zero:],
                             times=times[times >= -0.25],
                             test_times=probe_test_times,
                             correction='Bonferroni')

# plot GAT performance
fig = plot_gat_matrix(data=scores[:, zero:, zero:].mean(0),
                      times=times[times >= -0.25],
                      stats_dict=stats_dict,
                      mask=(sig_mask if bootstrap else None),
                      vmax=0.7, vmin=0.3,
                      draw_mask=True, draw_contour=True,
                      draw_diag=True, draw_zerolines=True,
                      xlabel="Time (s)", ylabel="Time (s)",
                      title_gat=('GAT (AX vs. BX)' if cue else 'GAT (AX vs. AY)'),
                      cmap="RdBu_r", mask_cmap="RdBu_r", mask_alpha=1.0,
                      test_times=probe_test_times,
                      focus=None,
                      legend_loc='upper center',
                      figsize=(9.0, 6.0))
fig.savefig('../results/figures/gat_matrix%s.png' % ('_probe' + contrast),
            dpi=600)

#%%
mean_scores = scores.mean(axis=0)
times[(times >= 1.0) & (times <= 1.2)]
acc_m = mean_scores.diagonal()[(times >= 1.0) & (times <= 1.2)]

sd_scores = scores.std(axis=0)
acc_sd = sd_scores.diagonal()[(times >= 1.0) & (times <= 1.2)]

acc_m - 2.576 * acc_sd / np.sqrt(52)

# %%
CPP = stats_dict['Cue CPP'][0]

tt = [0.6, 0.65]
t_times = times[times >= -0.25]
t_times[(t_times >= tt[0]) & (t_times <= tt[1])]
acc_m = CPP.mean(axis=0)[(t_times >= tt[0]) & (t_times <= tt[1])]

sd_scores = CPP.std(axis=0)
acc_sd = sd_scores[(t_times >= tt[0]) & (t_times <= tt[1])]

acc_m + 2.576 * acc_sd / np.sqrt(52)
