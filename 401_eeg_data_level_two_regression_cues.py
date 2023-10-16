"""
==============================
Compute group-level statistics
==============================

Compute T- and F- for the effect of conditions and search for
significant (spatio-temporal) clusters.

Authors: José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
# %%
# imports
import sys
import os
import glob

import json

import matplotlib.pyplot as plt

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

# create path for epochs import
str_subj = str(subject).rjust(3, '0')
# create path for beta coefficients
FPATH_BETAS = os.path.join(FPATH_DERIVATIVES,
                           'limo',
                           'sub-*',
                           'sub-*_betas.npy')
FPATH_BETAS = glob.glob(FPATH_BETAS)
FPATH_BETAS.sort()

# %%
# load subject level results
shape_like = np.load(FPATH_BETAS[0]).shape

betas = np.empty((len(FPATH_BETAS), shape_like[0], shape_like[1]))
for nsubj, fpath in enumerate(FPATH_BETAS):
    betas[nsubj, ...] = np.load(fpath)

# create path for preprocessed data
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_cues-epo.fif' % f'{subject:03}')
# get the data
cue_epo = read_epochs(FPATH_EPOCHS, preload=True).crop(
    tmin=-0.25, tmax=2.50, include_tmax=True)
epochs_info = cue_epo.info
tmin = cue_epo.tmin
times = cue_epo.times
n_times = times.shape[0]
channels = cue_epo.ch_names
n_channels = len(channels)
# shape of eeg data (channels x times)
eeg_shape = (n_channels, n_times)

# sensor adjacency matrix
adjacency, _ = find_ch_adjacency(epochs_info, ch_type='eeg')

# %%
# get condition specific beta-coefficients

# time resolved betas for the B-cue
cue_b_betas = betas[:, 1, :]
# time resolved betas for the A-cue
cue_a_betas = betas[:, 0, :]

# also save them in "EEG-format" (channels x times)
A_coef = np.zeros((betas.shape[0], n_channels, n_times))
B_coef = np.zeros((betas.shape[0], n_channels, n_times))
for sub in range(betas.shape[0]):
    A_coef[sub, :] = betas[sub, 0, :].reshape(eeg_shape)
    B_coef[sub, :] = betas[sub, 1, :].reshape(eeg_shape)

# %%
# compute observed t-values for Cue B - Cue A contrast
MI = ModelInference()
MI.paired_ttest(data_one=cue_b_betas,
                data_two=cue_a_betas,
                adjacency=None)

# extract test-values
t_vals = MI.t_vals_.copy()

# put them in mne.Evoked format
cue_contrast_t = EvokedArray(t_vals.copy().reshape((n_channels, n_times)),
                             cue_epo.info, tmin)

# %%
# run bootstrap for Cue B - Cue A contrast

# n samples
boot = 2000

# make path
boot_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'boottvals_contrast_cues.npy'
)
# check if the file already exists
if not os.path.isfile(boot_path):
    boot_tvals = bootstrap_ttest(data_one=cue_b_betas, data_two=cue_a_betas,
                                 nboot=boot,
                                 multcomp=False,
                                 random=True,
                                 jobs=jobs)
    np.save(boot_path, boot_tvals)
else:
    boot_tvals = np.load(boot_path)

# run bootstrap to control for multiple comparisons (FWE)
# find critical F-max distribution under H0
method = 'fmax'
fmax_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'sig_contrast_cues_%s.npy' % method
)
if not os.path.isfile(fmax_path):
    f_max = bootstrap_ttest(data_one=cue_b_betas, data_two=cue_a_betas,
                            nboot=boot,
                            multcomp=method,
                            random=True,
                            jobs=jobs)
    np.save(fmax_path, f_max)
else:
    f_max = np.load(fmax_path)

# plot f-max thresholds
plot_fmax = True
if plot_fmax:
    f_max.copy()
    y_max = np.histogram(f_max, bins=100)[0]
    fig_h0, ax = plt.subplots()
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

# compute CIS
# lower threshold
l_tval = np.quantile(boot_tvals, axis=0, q=0.05 / 2)
l_tval = l_tval.reshape((n_channels, n_times))
# upper threshold
u_tval = np.quantile(boot_tvals, axis=0, q=1 - 0.05 / 2)
u_tval = u_tval.reshape((n_channels, n_times))

# %%
# asses significance of t-values at p < 0.01 (FWE corrected)
sig_threshold = np.quantile(f_max, [.99], axis=0)
sig_mask = (t_vals.reshape((n_channels, n_times)) ** 2) >= sig_threshold

# %%
# plot results of second level analysis

# plot mass-univariate results
fig = plot_contrast_tvals(cue_contrast_t,
                          times=[0.210, 0.640, 1.180, 1.365],
                          mask=sig_mask,
                          xlim=[-0.25, 2.50],
                          clim=[-12, 12])
cue_contrast_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_cues_tvals_%s.png'
                               % method
)
fig.savefig(cue_contrast_fig_path, dpi=600)

# %%
# get peaks
early_positive = cue_contrast_t.get_peak(
    tmin=0.150, tmax=0.250, mode='pos', return_amplitude=True
)
early_negative = cue_contrast_t.get_peak(
    tmin=0.150, tmax=0.250, mode='neg', return_amplitude=True
)

midrange_positive = cue_contrast_t.get_peak(
    tmin=0.400, tmax=0.750, mode='pos', return_amplitude=True
)
midrange_negative = cue_contrast_t.get_peak(
    tmin=0.400, tmax=0.750, mode='neg', return_amplitude=True
)

late_positive = cue_contrast_t.get_peak(
    tmin=1.000, tmax=1.200, mode='pos', return_amplitude=True
)
late_negative = cue_contrast_t.get_peak(
    tmin=1.000, tmax=1.200, mode='neg', return_amplitude=True
)

later_positive = cue_contrast_t.get_peak(
    tmin=1.250, tmax=1.500, mode='pos', return_amplitude=True
)
later_negative = cue_contrast_t.get_peak(
    tmin=1.250, tmax=1.500, mode='neg', return_amplitude=True
)

# store in dictionary
cue_effects = {
    'early':
        {'positive': early_positive,
         'negative': early_negative},
    'midrange':
        {'positive': midrange_positive,
         'negative': midrange_negative},
    'late':
        {'positive': late_positive,
         'negative': late_negative},
    'later':
        {'positive': later_positive,
         'negative': later_negative},
}

# save cluster peaks
cue_contrast_peaks_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_cue_peaks_%s.json' % method
)
with open(cue_contrast_peaks_path, 'w') as fp:
    json.dump(cue_effects, fp)

# %%
# plot exemplary sensors
fig = plot_contrast_sensor(cue_contrast_t,
                           lower_b=l_tval, upper_b=u_tval,
                           sig_mask=sig_mask,
                           sensors=['P6', 'Pz', 'FC1'],
                           xlim=[-0.25, 2.50],
                           ylim=[-15, 15],
                           figsize=(5.5, 10.5),
                           panel_letters=['d', 'e', 'f'])
cue_contrast_sensors_fig_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'ttest_cues_sensors_%s.png'
                               % method
)
fig.savefig(cue_contrast_sensors_fig_path, dpi=600)

# %%
# test effect of behavioural a-cue bias on the amplitude response evoked
# by the A and B-cues

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
lm.fit(cue_b_betas - cue_a_betas, a_bias_design)

a_bias_betas = lm.coef_.copy()

# put them in mne.Evoked format
a_bias_effect = EvokedArray(
    a_bias_betas.copy().reshape((n_channels, n_times)) * 1e6,
    cue_epo.info, -0.25)

# extract random subjects from overall sample
a_bias_boot = np.zeros((boot, n_channels * n_times))
random = np.random.RandomState(42)
for i in range(0, boot):
    boot_samples = random.choice(
        range(betas.shape[0]), betas.shape[0], replace=True)

    lm = LinearModel()
    lm.fit(betas[boot_samples, 1, :] - betas[boot_samples, 0, :],
           a_bias_design.iloc[boot_samples])

    a_bias_boot[i, :] = lm.coef_.copy()

l_tval = np.quantile(a_bias_boot, axis=0, q=0.01 / 2)
l_tval = l_tval.reshape((n_channels, n_times))
u_tval = np.quantile(a_bias_boot, axis=0, q=1 - 0.01 / 2)
u_tval = u_tval.reshape((n_channels, n_times))

mask = ((l_tval > 0) & (u_tval > 0)) | ((u_tval < 0) & (l_tval < 0))

fig = plot_contrast_tvals(a_bias_effect,
                          times=[0.45, 1.32, 2.07],
                          mask=mask,
                          xlim=[-0.0, 2.50],
                          clim=[-1.5, 1.5])

fig = a_bias_effect.plot_joint(times=[1.12, 1.29],
                               topomap_args=dict(vlim=(-2.5, 2.5),
                                                 time_unit='ms'),
                               ts_args=dict(ylim=dict(eeg=[-2, 2]),
                                            time_unit='ms'))
fig.axes[0].set_ylabel(r'$\beta$ ($\mu$V)',
                       labelpad=10.0, fontsize=12.0)
fig.axes[0].set_xlabel('Time (ms)',
                       labelpad=10.0, fontsize=12.0)
fig.axes[0].axvline(x=0, ymin=-2.5, ymax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axhline(y=0, xmin=-0.25, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)

fig.axes[1].text(-2.0, 1.0, 'a' + ' |',
                 transform=fig.axes[1].transAxes,
                 fontsize='x-large', verticalalignment='top')
a_cue_bias_cue_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'a_bias_cue_evoked.png'
)
fig.savefig(a_cue_bias_cue_path, dpi=300)

fig = plot_contrast_sensor(a_bias_effect,
                           lower_b=l_tval, upper_b=u_tval,
                           sig_mask=None,
                           sensors=['O2', 'AF4'],
                           xlim=[-0.25, 2.5],
                           ylim=[-3, 3],
                           ylabel=r'$\beta$ ($\mu$V)',
                           panel_letters=['b', 'c'],
                           figsize=(7, 8),
                           scale=1e6)
a_cue_bias_cue_effect_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'a_cue_bias_cue_effect.png'
)
fig.savefig(a_cue_bias_cue_effect_path, dpi=300)

# %%
# test effect of behavioural d' context on the amplitude response evoked
# by the A and B-cues

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
lm.fit(cue_b_betas - cue_a_betas, d_context_design)

d_context_betas = lm.coef_.copy()

# put them in mne.Evoked format
d_context_effect = EvokedArray(
    d_context_betas.copy().reshape((n_channels, n_times)) * 1e6,
    cue_epo.info, -0.25)

# extract random subjects from overall sample
d_context_boot = np.zeros((boot, n_channels * n_times))
random = np.random.RandomState(42)
for i in range(0, boot):

    boot_samples = random.choice(
        range(betas.shape[0]), betas.shape[0], replace=True)
    lm = LinearModel()
    lm.fit(betas[boot_samples, 1, :] - betas[boot_samples, 0, :],
           d_context_design.iloc[boot_samples])

    d_context_boot[i, :] = lm.coef_

l_tval = np.quantile(d_context_boot, axis=0, q=0.01 / 2)
l_tval = l_tval.reshape((n_channels, n_times))
u_tval = np.quantile(d_context_boot, axis=0, q=1 - 0.01 / 2)
u_tval = u_tval.reshape((n_channels, n_times))

mask = ((l_tval > 0) & (u_tval > 0)) | ((u_tval < 0) & (l_tval < 0))

fig = plot_contrast_tvals(d_context_effect,
                          times=[0.45, 1.32, 2.07],
                          mask=mask,
                          xlim=[-0.0, 2.50],
                          clim=[-1.5, 1.5])

fig = d_context_effect.plot_joint(times=[0.55, 1.400, 2.14],
                                  topomap_args=dict(vlim=(-2.5, 2.5),
                                                    time_unit='ms'),
                                  ts_args=dict(ylim=dict(eeg=[-2, 2]),
                                               time_unit='ms'))
fig.axes[0].set_ylabel(r'$\beta$ ($\mu$V)',
                       labelpad=10.0, fontsize=12.0)
fig.axes[0].set_xlabel('Time (ms)',
                       labelpad=10.0, fontsize=12.0)
fig.axes[0].axvline(x=0, ymin=-2.5, ymax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)
fig.axes[0].axhline(y=0, xmin=-0.25, xmax=2.5,
                    color='black', linestyle='dashed', linewidth=.8)

fig.axes[1].text(-2.0, 1.0, 'a' + ' |',
                 transform=fig.axes[1].transAxes,
                 fontsize='x-large', verticalalignment='top')
d_context_cue_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'd_context_cue_evoked.png'
)
fig.savefig(d_context_cue_path, dpi=300)

fig = plot_contrast_sensor(d_context_effect,
                           lower_b=l_tval, upper_b=u_tval,
                           sig_mask=None,
                           sensors=['CP5', 'TP7'],
                           xlim=[-0.25, 2.5],
                           ylim=[-3, 3],
                           ylabel=r'$\beta$ ($\mu$V)',
                           panel_letters=['b', 'c'],
                           figsize=(7, 8),
                           scale=1e6)
d_context_cue_effect_path = os.path.join(
    FPATH_DERIVATIVES, 'limo', 'd_context_cue_effect.png'
)
fig.savefig(d_context_cue_effect_path, dpi=300)

pch, ptime = d_context_effect.get_peak(tmin=0.5, tmax=1.0, mode='pos')
print(l_tval[channels.index(pch), times == ptime] * 1e6)
print(u_tval[channels.index(pch), times == ptime] * 1e6)
