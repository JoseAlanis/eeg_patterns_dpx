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

from mne import (read_epochs, concatenate_epochs, EvokedArray,
                 open_report, combine_evoked)
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

# create path for preprocessed data
FPATH_EPOCHS = os.path.join(
    FPATH_DERIVATIVES,
    'epochs',
    'sub-%s' % f'{subject:03}',
    'sub-%s_cues-epo.fif' % f'{subject:03}'
)
# get the data
cue_epo = read_epochs(FPATH_EPOCHS, preload=True)

# %%
# pre-select epochs for fitting

# 1. remove correct responses that were preceded by
#    an error or "too slow" responses
if remove_pre_error:
    # shift data by one
    cue_epo.metadata['shifted_response'] = \
        cue_epo.metadata['reaction_probe'].shift(1)

    # remove epochs that preceded erroneous responses
    cue_epo = cue_epo['shifted_response != "Too_soon"']
    cue_epo = cue_epo['shifted_response != "Incorrect"']

# only keep correct responses
cue_epo = cue_epo['reaction_probe == "Correct"']

# remove epochs with RTs < 0.1 s and > 1.0 s
cue_epo = cue_epo['rt > 0.1 & rt < 1.0']

# 2. equalize number of errors
a_correct = cue_epo['Correct A']
b_correct = cue_epo['Correct B']
if equalize_count:
    # set random state for replication
    random_state = 42
    random = np.random.RandomState(random_state)

    # get correct B-Cue trials
    b_trials = b_correct.metadata.trial

    # get correct A-Cue trials
    a_trials = a_correct.metadata.trial

    # equalize trial counts
    a_trials = random.choice(a_trials, len(b_trials), replace=False)
    a_correct = a_correct[a_correct.metadata.trial.isin(a_trials)]

# %%
# prepare epochs for analysis

# set baseline and analysis time window
bstime = (-0.300, -0.050)

# keep time period around event
a_correct = a_correct.crop(tmin=-0.5, tmax=2.5).apply_baseline(baseline=bstime)
b_correct = b_correct.crop(tmin=-0.5, tmax=2.5).apply_baseline(baseline=bstime)

# save the generic info structure of cue epochs (i.e., channel names, number of
# channels, etc.).
epochs_info = a_correct.info
n_channels = len(epochs_info['ch_names'])
times = a_correct.times

# analysis time window = -0.25 to 2.50 s
tmin = -0.25
tmax = 2.50
analysis_times = (times >= tmin) & (times <= tmax)
n_times = len([t for t in analysis_times if t])

# %%
# choose predictors

# independent variables to be used in the analysis (i.e., predictors)
predictors = ['cue']

# %%
# run linear regression

logger.info(
    "\n\nRunning linear regression analysis (level 1)"
    "\nSubject:    %s"
    "\nPredictors: %s"
    "\n\n" % (subject, ', '.join(str(x) for x in predictors))
)

# ** get Y data (targets) **
y_epochs = concatenate_epochs([a_correct, b_correct])
y_dat = y_epochs.get_data()
# only keep time period for analysis
y_dat = y_dat[:, :, analysis_times]
# make it a 2-array (epochs x (channels * times))
Y = Vectorizer().fit_transform(y_dat)

# ** get X data (predictors) and create design matrix
metadata = y_epochs.metadata.copy()
design = metadata[predictors]
design = dmatrix("0 + cue", design, return_type='dataframe')

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
# 2 betas (cue A and cue B)
betas = np.zeros((2, (n_channels * n_times)))

# A-cue beta
betas[0, :] = coefs[0, :]
# B-cue beta
betas[1, :] = coefs[1, :]

# back projection to channels x time points
A = coefs[0, :].reshape((n_channels, n_times))
B = coefs[1, :].reshape((n_channels, n_times))

# compute global field power (GFP) effects
A_GFP = np.sqrt(((A * 1e6) ** 2).mean(axis=0))
B_GFP = np.sqrt(((B * 1e6) ** 2).mean(axis=0))

# put betas in an evoked object for easier handling
A_evoked = EvokedArray(A, epochs_info, tmin)
B_evoked = EvokedArray(B, epochs_info, tmin)
r2_evoked = EvokedArray(r2, epochs_info, tmin)

# %%
# get peaks beta coefficients for each cue across different time periods

gfps = {'A': A_GFP, 'B': B_GFP, '$R^2$': r2_gfp}
evokeds = {'A': A_evoked, 'B': B_evoked, '$R^2$': r2_evoked}

times_to_plot = [time for time, t in zip(times, analysis_times) if t]

viridis = colormaps['magma']
colors = np.linspace(0.15, 0.65, len(gfps)-1)
fig, ax = plt.subplots(2, 1, figsize=[7.5, 5])
for n, gfp in enumerate(gfps):
    gfp_vals = gfps[gfp]

    if gfp not in ['A', 'B']:
        gfp_vals = gfp_vals * 100
        ax[1].plot(times_to_plot, gfp_vals, label=r'%s GFP' % gfp,
                   color='black')
        ax[1].legend(loc='upper right', framealpha=0)
        r_sq_units = np.arange(0, round(max(gfp_vals))+0.1, 1)
        ax[1].set_yticks(r_sq_units)
        ax[1].set_yticklabels(r_sq_units / 100)
        ax[1].set_xticks(np.arange(-0.25, 2.51, 0.25))
        ax[1].axvline(x=0, ymin=-5, ymax=5,
                      color='black', linestyle='dashed', linewidth=1.0)
        ax[1].set_ylabel(r'GFP ($R^2$-units)')
        ax[1].set_xlabel('Time (s)')
    else:
        ax[0].plot(times_to_plot, gfp_vals, label='%s GFP' % gfp,
                   color=viridis(colors[n]))
        ax[0].legend(loc='lower right', framealpha=0)

    ax[0].set_xticks(np.arange(-0.25, 2.51, 0.25))
    ax[0].set_ylabel(r'GFP ($\mu$V)')
    ax[0].set_xlabel('Time (s)')
    ax[0].axvline(x=0, ymin=-5, ymax=5,
                  color='black', linestyle='dashed', linewidth=1.0)
    fig.subplots_adjust(hspace=0.5)
    plt.close('all')

beta_peaks = np.empty((1, 5))
for gfp in gfps:
    if gfp not in ['A', 'B']:
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
        if gfp not in ['A', 'B']:
            for mode in ['pos']:
                sensor, time, amplitude = evokeds[gfp].get_peak(
                    tmin=peak_time - 0.01 if peak_time > -0.24 else -0.25,
                    tmax=peak_time + 0.01 if peak_time < 2.49 else 2.50,
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
                    tmax=peak_time + 0.01 if peak_time < 2.49 else 2.50,
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
contrast_ba = combine_evoked([A_evoked, B_evoked], weights=[-1, 1])
fig_contrast = contrast_ba.plot_joint(times=[0.2, 0.3, 0.5, 0.6, 1.1, 2.3],
                                      show=False)
plt.close('all')

# %%
# save level-1 results

# create path for preprocessed dara
FPATH_BETAS = os.path.join(
    FPATH_DERIVATIVES,
    'limo',
    'sub-%s' % f'{subject:03}',
    'sub-%s_betas.npy' % f'{subject:03}'
)

# check if directory exists
if not Path(FPATH_BETAS).exists():
    Path(FPATH_BETAS).parent.mkdir(parents=True, exist_ok=True)

# save betas
np.save(FPATH_BETAS, betas)

# create path for preprocessed dara
FPATH_PEAKS = os.path.join(FPATH_DERIVATIVES,
                           'limo',
                           'sub-%s' % f'{subject:03}',
                           'sub-%s_peaks.tsv' % f'{subject:03}')

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
        title='GFP effects',
        image_format='PNG',
        replace=True
    )

    bidsdata_report.add_figure(
        fig=fig_contrast,
        tags='regression-analysis',
        title='Contrast B-A',
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
