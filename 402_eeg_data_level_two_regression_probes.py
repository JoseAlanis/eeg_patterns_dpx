"""
==============================
Compute group-level statistics
==============================

Compute T- and F- for the effect of conditions and search for
significant (spatio-temporal) clusters.

Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
# imports
import sys
import os
import glob

import re

import json

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import numpy as np
import pandas as pd

from scipy.stats import zscore

from mne import read_epochs, EvokedArray
from mne.utils import logger
from mne.channels import find_ch_adjacency

# All parameters are defined in config.py
from config import (
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS
)

from utils import parse_overwrite

from stats import LinearModel, ModelInference, bootstrap_ttest
from viz import plot_contrast_tvals, plot_contrast_sensor

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
        subject=subject,
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

# create path for beta coefficients
FPATH_BETAS = os.path.join(FPATH_DERIVATIVES,
                           'limo',
                           'sub-*',
                           'sub-*_betas_probes.npy')
FPATH_BETAS = glob.glob(FPATH_BETAS)

# %%
# load subject level results
shape_like = np.load(FPATH_BETAS[0]).shape

betas = np.empty((len(FPATH_BETAS), shape_like[0], shape_like[1]))
for fpath in FPATH_BETAS:
    sj = re.search(r'\d+', os.path.basename(fpath)).group(0)
    betas[int(sj) - 1, ...] = np.load(fpath)

# create path for preprocessed dara
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_probes-epo.fif' % f'{subject:03}')
# get the data
probe_epo = read_epochs(FPATH_EPOCHS, preload=True).crop(
    tmin=-0.25, tmax=1.00, include_tmax=True)
epochs_info = probe_epo.info
tmin = probe_epo.tmin
times = probe_epo.times
n_times = times.shape[0]
channels = probe_epo.ch_names
n_channels = len(channels)
# shape of eeg data (channels x times)
eeg_shape = (n_channels, n_times)

# sensor adjacency matrix
adjacency, _ = find_ch_adjacency(epochs_info, ch_type='eeg')

# %%
# get condition specific beta-coefficients

AX_coef = np.zeros((betas.shape[0], n_channels, n_times))
AY_coef = np.zeros((betas.shape[0], n_channels, n_times))
BX_coef = np.zeros((betas.shape[0], n_channels, n_times))
BY_coef = np.zeros((betas.shape[0], n_channels, n_times))

for sub in range(betas.shape[0]):
    AX_coef[sub, :] = betas[sub, 0, :].reshape(eeg_shape)
    AY_coef[sub, :] = betas[sub, 1, :].reshape(eeg_shape)
    BX_coef[sub, :] = betas[sub, 2, :].reshape(eeg_shape)
    BY_coef[sub, :] = betas[sub, 3, :].reshape(eeg_shape)

# %%
# predictors
preds = {'AX': 0, 'AY': 1, 'BX': 2, 'BY': 3}

# condition of interest
pred_0 = 'AX'
pred_1 = 'AY'

# condiotions for analysis
probe_0_coef = betas[:, preds[pred_0], :]
probe_1_coef = betas[:, preds[pred_1], :]

# compute observed t-values for Cue AY - Cue AX contrast
MI = ModelInference()
MI.paired_ttest(data_one=probe_1_coef,
                data_two=probe_0_coef,
                adjacency=None)

# extract test-values
t_vals = MI.t_vals_.copy()

# put them in mne.Evoked format
preds_contrast_t = EvokedArray(t_vals.copy().reshape((n_channels, n_times)),
                               probe_epo.info, tmin)

# %%
# run bootstrap for Cue B - Cue A contrast

# number of samples
boot = 2000

# make path
probe_contr = pred_1 + '_' + pred_0
boot_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'boottvals_contrast_probes_%s.npy' % probe_contr
)
if not os.path.isfile(boot_path):
    boot_tvals = bootstrap_ttest(data_one=probe_1_coef,
                                 data_two=probe_0_coef,
                                 nboot=boot,
                                 multcomp=False,
                                 random=True,
                                 jobs=jobs)
    np.save(boot_path, boot_tvals)
else:
    boot_tvals = np.load(boot_path)

# lower threshold
l_tval = np.quantile(boot_tvals, axis=0, q=0.01 / 2)
l_tval = l_tval.reshape((n_channels, n_times))
# upper threshold
u_tval = np.quantile(boot_tvals, axis=0, q=1 - 0.01 / 2)
u_tval = u_tval.reshape((n_channels, n_times))

# run bootstrap to control for multiple comparisons (FWE)
# find critical F-max distribution under H0
method = 'fmax'
fmax_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'sig_contrast_probes_%s_%s.npy'
                               % (probe_contr, method)
)
if not os.path.isfile(fmax_path):
    f_max = bootstrap_ttest(data_one=probe_1_coef,
                            data_two=probe_0_coef,
                            nboot=boot,
                            multcomp=method,
                            jobs=jobs)
    np.save(fmax_path, f_max)
else:
    f_max = np.load(fmax_path)

# plot f-max thresholds
plot_fmax = True
if plot_fmax:
    f_max.copy()
    y_max = np.histogram(f_max, bins=100)[0]
    fig, ax = plt.subplots()
    ax.hist(f_max, ec='k', bins=100)
    ax.axvline(np.quantile(f_max, 0.95),
               color='red', linestyle='dashed', linewidth=2)
    ax.text(x=np.quantile(f_max, 0.95),
            y=np.max(y_max),
            s='p<0.05',
            fontdict=dict(fontsize=12),
            va='top',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
    ax.axvline(np.quantile(f_max, 0.99),
               color='red', linestyle='dashdot', linewidth=2)
    ax.text(x=np.quantile(f_max, 0.99),
            y=np.max(y_max)/2,
            s='p<0.01',
            fontdict=dict(fontsize=12),
            va='top',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))

# %%
# asses significance of t-values at p < 0.01 (FWE corrected)
sig_threshold = np.quantile(f_max, [.99], axis=0)
sig_mask = t_vals.reshape((n_channels, n_times)) ** 2
sig_mask = sig_mask > sig_threshold

# %%
# plot results of second level analysis

# plot mass-univariate results
fig = plot_contrast_tvals(preds_contrast_t,
                          figsize=(5.5, 5.0),
                          times=[0.22, 0.360, 0.550],
                          mask=sig_mask,
                          xlim=[-0.25, 1.00],
                          clim=[-12, 12])
probe_contrast_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_probes_%s_tvals_%s.tiff'
                               % (probe_contr, method)
)
fig.savefig(probe_contrast_fig_path, dpi=600)

# %%
# get peaks
early_positive = preds_contrast_t.get_peak(
    tmin=0.150, tmax=0.250, mode='pos', return_amplitude=True
)
early_negative = preds_contrast_t.get_peak(
    tmin=0.150, tmax=0.250, mode='neg', return_amplitude=True
)

midrange_positive = preds_contrast_t.get_peak(
    tmin=0.250, tmax=0.400, mode='pos', return_amplitude=True
)
midrange_negative = preds_contrast_t.get_peak(
    tmin=0.250, tmax=0.400, mode='neg', return_amplitude=True
)

late_positive = preds_contrast_t.get_peak(
    tmin=0.500, tmax=0.750, mode='pos', return_amplitude=True
)
late_negative = preds_contrast_t.get_peak(
    tmin=0.500, tmax=0.750, mode='neg', return_amplitude=True
)

# store in dictionary
probe_effects = {
    'early':
        {'positive': early_positive,
         'negative': early_negative},
    'midrange':
        {'positive': midrange_positive,
         'negative': midrange_negative},
    'late':
        {'positive': late_positive,
         'negative': late_negative},
}

# save cluster peaks
probe_contrast_peaks_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_probes_%s_peaks_%s.json'
                               % (probe_contr, method)
)
with open(probe_contrast_peaks_path, 'w') as fp:
    json.dump(probe_effects, fp)

# plot exemplary sensors
label = probe_contr.upper().split('_')
label = label[0] + ' - ' + label[1]
fig = plot_contrast_sensor(preds_contrast_t,
                           lower_b=l_tval,
                           upper_b=u_tval,
                           sig_mask=sig_mask,
                           sensors=['P6', 'Pz', 'CPz'],
                           xlim=[-0.25, 1.0],
                           ylim=[-15, 15],
                           figsize=(5.5, 10.5),
                           legend_fontsize='small',
                           panel_letters=['f', 'g', 'h'],
                           label=label)
probe_contrast_sensors_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_probes_sensors_%s_%s.png'
                               % (probe_contr, method)
)
fig.savefig(probe_contrast_sensors_fig_path, dpi=600)

# %%
# test effect of behavioural a-cue bias on the amplitude response evoked
# by the X and Y-cues

# import subject covariates
a_bias = pd.read_csv(os.path.join(FPATH_DERIVATIVES,
                                  'behavioural_analysis',
                                  '02_a_bias.tsv'),
                     sep='\t',
                     header=0)
a_bias_design = a_bias.assign(intercept=1)
a_bias_design = a_bias_design.assign(a_bias=zscore(a_bias_design.a_bias))
a_bias_design = a_bias_design[['a_bias']]

lm = LinearModel()
lm.fit(probe_1_coef - probe_0_coef, a_bias_design)

a_bias_betas = lm.coef_

# put them in mne.Evoked format
a_bias_effect = EvokedArray(
    a_bias_betas.copy().reshape((n_channels, n_times)) * 1e6,
    probe_epo.info, -0.25)

# extract random subjects from overall sample
a_bias_boot = np.zeros((boot, n_channels * n_times))
random = np.random.RandomState(42)
for i in range(0, boot):
    boot_samples = random.choice(
        range(betas.shape[0]), betas.shape[0], replace=True)

    lm = LinearModel()
    lm.fit(probe_1_coef[boot_samples, :] - probe_0_coef[boot_samples, :],
           a_bias_design.iloc[boot_samples])

    a_bias_boot[i, :] = lm.coef_.copy()

a = int(((0.01 * boot) / (2 * 1)) + 1)
a_bias_boot.sort(axis=0)
l_tval = a_bias_boot[a, :]
u_tval = a_bias_boot[int(boot - a), :]
l_tval = l_tval.reshape((n_channels, n_times))
u_tval = u_tval.reshape((n_channels, n_times))

mask = ((l_tval > 0) & (u_tval > 0)) | ((u_tval < 0) & (l_tval < 0))
fig = plot_contrast_tvals(a_bias_effect,
                          times=[0.090, 0.26, 0.30, 0.70, 0.85],
                          mask=mask,
                          xlim=[-0.0, 1.0],
                          clim=[-1.5, 1.5])
a_bias_probe_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'a_bias_probes_%s_effect_%s.png'
                               % (probe_contr, method)
)
fig.savefig(a_bias_probe_fig_path, dpi=600)

# get peaks
early_positive = a_bias_effect.get_peak(
    tmin=0.090, tmax=0.150, mode='pos', return_amplitude=True
)
early_negative = a_bias_effect.get_peak(
    tmin=0.090, tmax=0.150, mode='neg', return_amplitude=True
)

# get peaks
mid_positive = a_bias_effect.get_peak(
    tmin=0.250, tmax=0.350, mode='pos', return_amplitude=True
)
mid_negative = a_bias_effect.get_peak(
    tmin=0.250, tmax=0.350, mode='neg', return_amplitude=True
)

# get peaks
late_positive = a_bias_effect.get_peak(
    tmin=0.65, tmax=0.75, mode='pos', return_amplitude=True
)
late_negative = a_bias_effect.get_peak(
    tmin=0.65, tmax=0.75, mode='neg', return_amplitude=True
)


# store in dictionary
a_cue_bias_effects = {
    'early':
        {'positive': early_positive,
         'pos_lCI': float(l_tval[channels.index(early_positive[0]), times == early_positive[1]] * 1e6),
         'pos_uCI': float(u_tval[channels.index(early_positive[0]), times == early_positive[1]] * 1e6),
         'negative': early_negative,
         'neg_lCI': float(l_tval[channels.index(early_negative[0]), times == early_negative[1]] * 1e6),
         'neg_uCI': float(u_tval[channels.index(early_negative[0]), times == early_negative[1]] * 1e6)},
    'mid':
        {'positive': mid_positive,
         'pos_lCI': float(l_tval[channels.index(mid_positive[0]), times == mid_positive[1]] * 1e6),
         'pos_uCI': float(u_tval[channels.index(mid_positive[0]), times == mid_positive[1]] * 1e6),
         'negative': mid_negative,
         'neg_lCI': float(l_tval[channels.index(mid_negative[0]), times == mid_negative[1]] * 1e6),
         'neg_uCI': float(u_tval[channels.index(mid_negative[0]), times == mid_negative[1]] * 1e6)},
    'late':
        {'positive': late_positive,
         'pos_lCI': float(l_tval[channels.index(late_positive[0]), times == late_positive[1]] * 1e6),
         'pos_uCI': float(u_tval[channels.index(late_positive[0]), times == late_positive[1]] * 1e6),
         'negative': late_negative,
         'neg_lCI': float(l_tval[channels.index(late_negative[0]), times == late_negative[1]] * 1e6),
         'neg_uCI': float(u_tval[channels.index(late_negative[0]), times == late_negative[1]] * 1e6)},
}

# save cluster peaks
a_cue_bias_probe_peaks_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'a_cue_bias_probe_%s_peaks_%s.json'
                               % (probe_contr, method)
)
with open(a_cue_bias_probe_peaks_path, 'w') as fp:
    json.dump(a_cue_bias_effects, fp)


# %%

# import subject covariates
d_context = pd.read_csv(os.path.join(FPATH_DERIVATIVES,
                                     'behavioural_analysis',
                                     '02_d_context.tsv'),
                        sep='\t',
                        header=0)
d_context_design = d_context.assign(intercept=1)
d_context_design = d_context_design.assign(
    d_context=zscore(d_context_design.d_context))
d_context_design = d_context_design[['d_context']]

lm = LinearModel()
lm.fit(probe_1_coef - probe_0_coef, d_context_design)

d_context_betas = lm.coef_

# put them in mne.Evoked format
d_context_effect = EvokedArray(
    d_context_betas.copy().reshape((n_channels, n_times)) * 1e6,
    probe_epo.info, -0.25)

# extract random subjects from overall sample
d_context_boot = np.zeros((boot, n_channels * n_times))
random = np.random.RandomState(42)
for i in range(0, boot):
    boot_samples = random.choice(
        range(betas.shape[0]), betas.shape[0], replace=True)

    lm = LinearModel()
    lm.fit(probe_1_coef[boot_samples, :] - probe_0_coef[boot_samples, :],
           d_context_design.iloc[boot_samples])

    d_context_boot[i, :] = lm.coef_.copy()

a = int(((0.01 * boot) / (2 * 1)) + 1)
d_context_boot.sort(axis=0)
l_tval = d_context_boot[a, :]
u_tval = d_context_boot[int(boot - a), :]
l_tval = l_tval.reshape((n_channels, n_times))
u_tval = u_tval.reshape((n_channels, n_times))

mask = ((l_tval > 0) & (u_tval > 0)) | ((u_tval < 0) & (l_tval < 0))
fig = plot_contrast_tvals(d_context_effect,
                          times=[0.20, 0.41, 0.52],
                          mask=mask,
                          xlim=[-0.0, 1.0],
                          clim=[-1.5, 1.5])
d_context_probe_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'd_context_probes_%s_effect_%s.png'
                               % (probe_contr, method)
)
fig.savefig(d_context_probe_fig_path, dpi=600)

# get peaks
early_positive = d_context_effect.get_peak(
    tmin=0.2, tmax=0.21, mode='pos', return_amplitude=True
)
early_negative = d_context_effect.get_peak(
    tmin=0.2, tmax=0.21, mode='neg', return_amplitude=True
)

# get peaks
mid_positive = d_context_effect.get_peak(
    tmin=0.40, tmax=0.550, mode='pos', return_amplitude=True
)
mid_negative = d_context_effect.get_peak(
    tmin=0.40, tmax=0.550, mode='neg', return_amplitude=True
)

# store in dictionary
d_context_effects = {
    'early':
        {'positive': early_positive,
         'pos_lCI': float(l_tval[channels.index(early_positive[0]), times == early_positive[1]] * 1e6),
         'pos_uCI': float(u_tval[channels.index(early_positive[0]), times == early_positive[1]] * 1e6),
         'negative': early_negative,
         'neg_lCI': float(l_tval[channels.index(early_negative[0]), times == early_negative[1]] * 1e6),
         'neg_uCI': float(u_tval[channels.index(early_negative[0]), times == early_negative[1]] * 1e6)},
    'mid':
        {'positive': mid_positive,
         'pos_lCI': float(l_tval[channels.index(mid_positive[0]), times == mid_positive[1]] * 1e6),
         'pos_uCI': float(u_tval[channels.index(mid_positive[0]), times == mid_positive[1]] * 1e6),
         'negative': mid_negative,
         'neg_lCI': float(l_tval[channels.index(mid_negative[0]), times == mid_negative[1]] * 1e6),
         'neg_uCI': float(u_tval[channels.index(mid_negative[0]), times == mid_negative[1]] * 1e6)},
}

# save cluster peaks
d_context_probe_peaks_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'd_context_probe_%s_peaks_%s.json'
                               % (probe_contr, method)
)
with open(d_context_probe_peaks_path, 'w') as fp:
    json.dump(d_context_effects, fp)
