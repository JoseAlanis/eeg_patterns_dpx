"""
==========================================================================
Fit single subject linear model to voltage evoked by stimulus presentation
==========================================================================

Mass-univariate analysis voltage data.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps

from scipy.signal import find_peaks

from patsy import dmatrix  # noqa

from mne import read_epochs, concatenate_epochs, EvokedArray, \
    open_report, combine_evoked
from mne.decoding import Vectorizer
from mne.utils import logger

from sklearn.metrics import r2_score

from stats import LinearModel

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
weights = False

# should reactions that preceded an error be excluded?
remove_pre_error = True
# should number of epochs be equalized for all conditions
equalize_count = True

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

if overwrite:
    logger.info("`overwrite` is set to ``True`` ")


# %%
# get eeg epochs

# create path for preprocessed dara
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_probes-epo.fif' % f'{subject:03}')
# get the data
probe_epo = read_epochs(FPATH_EPOCHS, preload=True)

# %%
# pre-select epochs for fitting

# 1. remove correct responses that were preceded by
#    an error or "too slow" responses
if remove_pre_error:
    # shift data by one
    probe_epo.metadata['shifted_response'] = \
        probe_epo.metadata['reaction_probe'].shift(1)

    # remove epochs that preceded erroneous responses
    probe_epo = probe_epo['shifted_response != "Too_soon"']
    probe_epo = probe_epo['shifted_response != "Incorrect"']

# only keep correct responses
probe_epo = probe_epo['reaction_probe == "Correct"']

# remove epochs with RTs < 0.1 s and > 1.0 s
probe_epo = probe_epo['rt > 0.1 & rt < 1.0']

# 2. equalize number of errors
ax_correct = probe_epo['Correct AX']
ay_correct = probe_epo['Correct AY']
bx_correct = probe_epo['Correct BX']
by_correct = probe_epo['Correct BY']
if equalize_count:
    # set random state for replication
    random_state = 42
    random = np.random.RandomState(random_state)

    # get correct B-Cue trials
    ax_trials = ax_correct.metadata.trial
    ay_trials = ay_correct.metadata.trial
    bx_trials = bx_correct.metadata.trial
    by_trials = by_correct.metadata.trial

    min_trials = np.min(
        [len(ax_trials), len(ay_trials), len(bx_trials), len(by_trials)]
    )

    # equalize trial counts
    ax_trials = random.choice(ax_trials, min_trials, replace=False)
    ay_trials = random.choice(ay_trials, min_trials, replace=False)
    bx_trials = random.choice(bx_trials, min_trials, replace=False)
    by_trials = random.choice(by_trials, min_trials, replace=False)

    ax_correct = ax_correct[ax_correct.metadata.trial.isin(ax_trials)]
    ay_correct = ay_correct[ay_correct.metadata.trial.isin(ay_trials)]
    bx_correct = bx_correct[bx_correct.metadata.trial.isin(bx_trials)]
    by_correct = by_correct[by_correct.metadata.trial.isin(by_trials)]

# %%
# prepare epochs for analysis

# set baseline and analysis time window
bstime = (-0.300, -0.050)

# keep time period around event
ax_correct = ax_correct.crop(tmin=-0.5, tmax=1.0).apply_baseline(baseline=bstime)  # noqa
ay_correct = ay_correct.crop(tmin=-0.5, tmax=1.0).apply_baseline(baseline=bstime)  # noqa
bx_correct = bx_correct.crop(tmin=-0.5, tmax=1.0).apply_baseline(baseline=bstime)  # noqa
by_correct = by_correct.crop(tmin=-0.5, tmax=1.0).apply_baseline(baseline=bstime)  # noqa

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.).
epochs_info = ax_correct.info
n_channels = len(epochs_info['ch_names'])
times = ax_correct.times

# analysis time window = -0.25 to 2.50 s
tmin = -0.25
tmax = 1.0
analysis_times = (times >= tmin) & (times <= tmax)
n_times = len([t for t in analysis_times if t])

# %%
# choose predictors

# independent variables to be used in the analysis (i.e., predictors)
predictors = ['cue_probe_combination']

# %%
# run linear regression

logger.info(
    "\n\nRunning linear regression analysis (level 1)"
    "\nSubject:    %s"
    "\nPredictors: %s"
    "\n\n" % (subject, ', '.join(str(x) for x in predictors))
)

# ** get Y data (targets) **
y_epochs = concatenate_epochs([ax_correct, ay_correct, bx_correct, by_correct])
y_dat = y_epochs.get_data()
# only keep time period for analysis
y_dat = y_dat[:, :, analysis_times]
# make it a 2-array (epochs x (channels * times))
Y = Vectorizer().fit_transform(y_dat)

# ** get X data (predictors) and create design matrix
metadata = y_epochs.metadata.copy()
design = metadata[predictors]
design = dmatrix("0 + cue_probe_combination", design, return_type='dataframe')

# compute beta coefficients and model predictions
linreg = LinearModel()
linreg.fit(Y, design)
coefs = linreg.coef_
y_hat = linreg.predict(design)

# %%
# compute R-squared (model fit)
r2 = r2_score(Y, y_hat,
              multioutput='raw_values')

r2 = r2.reshape((n_channels, n_times))

# compute global field power (GFP) for model fit
r2_gfp = np.sqrt((r2 ** 2).mean(axis=0))

# %%
# extract results

# initialise placeholders for the storage of results
# 2 betas (AX, AY, BX, and BY)
betas = np.zeros((4, (n_channels * n_times)))

# AX-probe beta
betas[0, :] = coefs[0, :]
# AY-probe beta
betas[1, :] = coefs[1, :]
# BX-probe beta
betas[2, :] = coefs[2, :]
# BY-probe beta
betas[3, :] = coefs[3, :]

# back projection to channels x time points
AX = coefs[0, :].reshape((n_channels, n_times))
AY = coefs[1, :].reshape((n_channels, n_times))
BX = coefs[2, :].reshape((n_channels, n_times))
BY = coefs[3, :].reshape((n_channels, n_times))

# compute global field power (GFP) effects
AX_GFP = np.sqrt(((AX * 1e6) ** 2).mean(axis=0))
AY_GFP = np.sqrt(((AY * 1e6) ** 2).mean(axis=0))
BX_GFP = np.sqrt(((BX * 1e6) ** 2).mean(axis=0))
BY_GFP = np.sqrt(((BY * 1e6) ** 2).mean(axis=0))

# put betas in an evoked object for easier handling
AX_evoked = EvokedArray(AX, epochs_info, tmin)
AY_evoked = EvokedArray(AY, epochs_info, tmin)
BX_evoked = EvokedArray(BX, epochs_info, tmin)
BY_evoked = EvokedArray(BY, epochs_info, tmin)
r2_evoked = EvokedArray(r2, epochs_info, tmin)

# %%
# get peaks beta coefficients for each cue across different time periods

gfps = {'AX': AX_GFP, 'AY': AY_GFP, 'BX': BX_GFP, 'BY': BY_GFP, '$R^2$': r2_gfp}
evokeds = {'AX': AX_evoked, 'AY': AY_evoked, 'BX': BX_evoked, 'BY': BY_evoked,
           '$R^2$': r2_evoked}

times_to_plot = [time for time, t in zip(times, analysis_times) if t]

viridis = colormaps['magma']
colors = np.linspace(0.15, 0.85, len(gfps)-1)
fig, ax = plt.subplots(2, 1, figsize=[7.5, 5])
for n, gfp in enumerate(gfps):
    gfp_vals = gfps[gfp]

    if gfp not in ['AX', 'AY', 'BX', 'BY']:
        gfp_vals = gfp_vals * 100
        ax[1].plot(times_to_plot, gfp_vals, label=r'%s GFP' % gfp,
                   color='black')
        ax[1].legend(loc='upper left', framealpha=0)
        r_sq_units = np.arange(0, round(max(gfp_vals))+0.1, 1)
        ax[1].set_yticks(r_sq_units)
        ax[1].set_yticklabels(r_sq_units / 100)
        ax[1].set_xticks(np.arange(-0.25, 1.01, 0.25))
        ax[1].axvline(x=0, ymin=-5, ymax=5,
                      color='black', linestyle='dashed', linewidth=1.0)
        ax[1].set_ylabel(r'GFP ($R^2$-units)')
        ax[1].set_xlabel('Time (s)')
    else:
        ax[0].plot(times_to_plot, gfp_vals, label='%s GFP' % gfp,
                   color=viridis(colors[n]))
        ax[0].legend(loc='upper left', framealpha=0)

    ax[0].set_xticks(np.arange(-0.25, 1.0, 0.25))
    ax[0].set_ylabel(r'GFP ($\mu$V)')
    ax[0].set_xlabel('Time (s)')
    ax[0].axvline(x=0, ymin=-5, ymax=5,
                  color='black', linestyle='dashed', linewidth=1.0)
    fig.subplots_adjust(hspace=0.5)
    plt.close('all')

beta_peaks = np.empty((1, 5))
for gfp in gfps:
    if gfp not in ['AX', 'AY', 'BX', 'BY']:
        prominence = 0.01
    else:
        prominence = 0.5
    peaks, properties = find_peaks(gfps[gfp],
                                   prominence=(prominence, None),
                                   width=1)

    for n_peak, peak in enumerate(peaks):
        peak_time = times_to_plot[peak]
        if peak_time < 0.0:
            continue

        width = properties['widths'][n_peak]
        if gfp not in ['AX', 'AY', 'BX', 'BY']:
            for mode in ['pos']:
                sensor, time, amplitude = evokeds[gfp].get_peak(
                    tmin=peak_time - 0.01 if peak_time > -0.24 else -0.25,
                    tmax=peak_time + 0.01 if peak_time < 0.99 else 1.0,
                    mode=mode,
                    return_amplitude=True)
                peak_stats = np.asarray(
                    [[gfp, sensor, time, width, round(amplitude * 1e6, 3)]]
                )

                beta_peaks = np.concatenate((beta_peaks, peak_stats))
        else:
            for mode in ['pos', 'neg']:
                sensor, time, amplitude = evokeds[gfp].get_peak(
                    tmin=peak_time - 0.01 if peak_time > -0.24 else -0.25,
                    tmax=peak_time + 0.01 if peak_time < 0.99 else 1.0,
                    mode=mode,
                    return_amplitude=True)
                peak_stats = np.asarray(
                    [[gfp, sensor, time, width, round(amplitude * 1e6, 3)]]
                )

                beta_peaks = np.concatenate((beta_peaks, peak_stats))

beta_peaks = beta_peaks[1:, :]
beta_peaks = pd.DataFrame(beta_peaks,
                          columns=['condition', 'sensor', 'time',
                                   'width', 'ampliutude'])

# %%
contrast_ay_ax = combine_evoked([AX_evoked, AY_evoked], weights=[-1, 1])
fig_contrast_ay_ax = contrast_ay_ax.plot_joint(
    times=[0.20, 0.25, 0.30, 0.35, 0.40],
    show=False)
plt.close('all')

contrast_ay_bx = combine_evoked([BX_evoked, AY_evoked], weights=[-1, 1])
fig_contrast_ay_bx = contrast_ay_bx.plot_joint(
    times=[0.20, 0.25, 0.30, 0.35, 0.40],
    show=False)
plt.close('all')


# %%
# save level-1 results

# create path for preprocessed dara
FPATH_BETAS = os.path.join(FPATH_DERIVATIVES,
                           'limo',
                           'sub-%s' % f'{subject:03}',
                           'sub-%s_betas_probes.npy' % f'{subject:03}')

# check if directory exists
if not Path(FPATH_BETAS).exists():
    Path(FPATH_BETAS).parent.mkdir(parents=True, exist_ok=True)

# save betas
np.save(FPATH_BETAS, betas)

# create path for preprocessed dara
FPATH_PEAKS = os.path.join(FPATH_DERIVATIVES,
                           'limo',
                           'sub-%s' % f'{subject:03}',
                           'sub-%s_probes_peaks.tsv'  % f'{subject:03}')

# check if directory exists
if not Path(FPATH_PEAKS).exists():
    Path(FPATH_PEAKS).parent.mkdir(parents=True, exist_ok=True)

# save rt data to disk
beta_peaks.to_csv(FPATH_PEAKS,
                  sep='\t',
                  index=False,
                  )

# %%
if report:
    # make path
    FPATH_REPORT = os.path.join(
        FPATH_DERIVATIVES,
        'preprocessing',
        'sub-%s' % f'{subject:03}',
        'report')

    FPATH_REPORT_I = os.path.join(
        FPATH_REPORT,
        'sub-%s_task-%s_preprocessing_report.hdf5'
        % (f'{subject:03}', 'dpx'))

    bidsdata_report = open_report(FPATH_REPORT_I)

    bidsdata_report.add_figure(
        fig=fig,
        tags='regression-analysis',
        title='GFP effects (probe)',
        image_format='PNG',
        replace=True
    )

    bidsdata_report.add_figure(
        fig=fig_contrast_ay_ax,
        tags='regression-analysis',
        title='Contrast AY-AX',
        image_format='PNG',
        replace=True
    )

    bidsdata_report.add_figure(
        fig=fig_contrast_ay_bx,
        tags='regression-analysis',
        title='Contrast AY-BX',
        image_format='PNG',
        replace=True
    )

    if overwrite:
        logger.info("`overwrite` is set to ``True`` ")

    for rep_ext in ['hdf5', 'html']:
        FPATH_REPORT_O = os.path.join(
            FPATH_REPORT,
            'sub-%s_task-%s_preprocessing_report.%s'
            % (f'{subject:03}', 'dpx', rep_ext))

        bidsdata_report.save(FPATH_REPORT_O,
                             overwrite=overwrite,
                             open_browser=False)
