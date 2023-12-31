"""
=============
Preprocessing
==============

Extracts relevant data and removes artefacts.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os

import warnings

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mne import events_from_annotations, concatenate_raws, Report
from mne.filter import notch_filter
from mne.preprocessing import ICA
from mne.utils import logger

from mne_bids import BIDSPath, read_raw_bids

from config import (
    FPATH_BIDS,
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS,
    event_id,
    noisy_channels
)

from utils import parse_overwrite

from pyprep.prep_pipeline import NoisyChannels
from mne_icalabel import label_components

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
    jobs = defaults["jobs"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(f"{subject} is not a valid subject ID.\n"
                     f"Use one of: {SUBJECT_IDS}")

if not os.path.exists(FPATH_BIDS):
    raise RuntimeError(
        MISSING_FPATH_BIDS_MSG.format(FPATH_BIDS)
    )

# %%
# create bids path for import
FNAME = BIDSPath(root=FPATH_BIDS,
                 subject=f'{subject:03}',
                 task='dpx',
                 datatype='eeg',
                 extension='.bdf')

if not os.path.exists(FNAME):
    warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
    sys.exit()

# %%
# get the data
raw = read_raw_bids(FNAME)
raw.load_data()

bad_recording = noisy_channels[subject]
if bad_recording == 'all':
    logger.info("Subject %s is bad, skipping." % subject)
    sys.exit()
raw.info['bads'] = bad_recording
raw.interpolate_bads(mode='accurate')
raw.set_eeg_reference('average')

# get sampling rate
sfreq = raw.info['sfreq']

# remove line noise
line_freq = [50.0, 100.0]
raw._data = notch_filter(raw.get_data(),
                         Fs=sfreq,
                         freqs=line_freq,
                         method="spectrum_fit",
                         mt_bandwidth=2,
                         p_value=0.05,
                         filter_length="10s",
                         n_jobs=jobs)

# fig, ax = plt.subplots(1,1)
# raw.compute_psd(n_jobs=jobs).plot(axes=ax)
# ax.set_title('Subejct %s' % subject)

# %%
# extract relevant parts of the recording

# eeg events
events = events_from_annotations(raw, event_id=event_id)

# extract cue events
cue_evs = events[0]
cue_evs = cue_evs[(cue_evs[:, 2] >= 70) & (cue_evs[:, 2] <= 75)]

# latencies and difference between two consecutive cues
latencies = cue_evs[:, 0] / sfreq
diffs = [(y - x) for x, y in zip(latencies, latencies[1:])]

# get first event after a long break (i.e., when the time difference between
# stimuli is greater than 10 seconds). This should only be the case in between
# task blocks
breaks = [diff for diff in range(len(diffs)) if diffs[diff] > 10]
logger.info("\nIdentified breaks at positions:\n %s " % ', '.join(
    [str(br) for br in breaks]))

# save start and end points of task blocks
# subject '041' has more practice trials (two rounds)
if subject == 41:
    # start of first block
    b1s = latencies[breaks[2] + 1] - 2
    # end of first block
    b1e = latencies[breaks[3]] + 6

    # start of second block
    b2s = latencies[breaks[3] + 1] - 2
    # end of second block
    b2e = latencies[breaks[4]] + 6

# all other subjects have the same structure
else:
    # start of first block
    b1s = latencies[breaks[0] + 1] - 2
    # end of first block
    b1e = latencies[breaks[1]] + 6

    # start of second block
    b2s = latencies[breaks[1] + 1] - 2
    # end of second block
    if len(breaks) > 2:
        b2e = latencies[breaks[2]] + 6
    else:
        b2e = latencies[-1] + 6

# %%
# extract data chunks belonging to the task blocks and concatenate them
# block 1
raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
# block 2
raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)
# concatenate
raw_bl = concatenate_raws([raw_bl1, raw_bl2])
del raw, raw_bl1, raw_bl2

# %%
# bad channel detection and interpolation

# make a copy of the data in question
raw_copy = raw_bl.copy()

# apply a 100Hz low-pass filter to data
raw_copy = raw_copy.filter(l_freq=None, h_freq=100.0,
                           picks=['eeg', 'eog'],
                           filter_length='auto',
                           l_trans_bandwidth='auto',
                           h_trans_bandwidth='auto',
                           method='fir',
                           phase='zero',
                           fir_window='hamming',
                           fir_design='firwin',
                           n_jobs=jobs)

# find bad channels
noisy_dectector = NoisyChannels(raw_copy, random_state=42, do_detrend=True)
noisy_dectector.find_all_bads(ransac=False)

# crate summary for PyPrep output
bad_channels = {'bads_by_deviation:': noisy_dectector.bad_by_deviation,
                'bads_by_hf_noise:': noisy_dectector.bad_by_hf_noise,
                'bads_by_correlation:': noisy_dectector.bad_by_correlation,
                'bads_by_SNR:': noisy_dectector.bad_by_SNR}

# interpolate the identified bad channels
raw_bl.info['bads'] = noisy_dectector.get_bads()
raw_bl.interpolate_bads(mode='accurate')

# %%
# prepare ICA

# set eeg reference
# raw_bl = raw_bl.set_eeg_reference('average')

# set ICA parameters
method = 'infomax'
reject = dict(eeg=250e-6)
ica = ICA(n_components=0.95,
          method=method,
          fit_params=dict(extended=True))

# make copy of raw with 1Hz high-pass filter
raw_4_ica = raw_bl.copy().filter(l_freq=1.0, h_freq=100.0, n_jobs=jobs)

# run ICA
ica.fit(raw_4_ica,
        reject=reject,
        reject_by_annotation=True)

# %%
# find bad components using ICA label
ic_labels = label_components(raw_4_ica, ica, method="iclabel")

labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in
               enumerate(labels) if label not in ["brain"]]

logger.info(f"Excluding these ICA components: {exclude_idx}")

# exclude the identified components and reconstruct eeg signal
ica.exclude = exclude_idx
ica.apply(raw_bl)

# clean up
del raw_4_ica

# %%
# apply filter to data
raw_bl = raw_bl.filter(l_freq=0.01, h_freq=30.0,
                       picks=['eeg', 'eog'],
                       filter_length='auto',
                       l_trans_bandwidth='auto',
                       h_trans_bandwidth='auto',
                       method='fir',
                       phase='zero',
                       fir_window='hamming',
                       fir_design='firwin',
                       n_jobs=jobs)

# %%
# create path for preprocessed data
FPATH_PREPROCESSED = os.path.join(
    FPATH_DERIVATIVES,
    'preprocessing',
    'sub-%s' % f'{subject:03}',
    'eeg',
    'sub-%s_task-%s_preprocessed-raw.fif' % (f'{subject:03}', 'dpx'))

# check if directory exists
if not Path(FPATH_PREPROCESSED).exists():
    Path(FPATH_PREPROCESSED).parent.mkdir(parents=True, exist_ok=True)

# save file
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

raw_bl.save(FPATH_PREPROCESSED, overwrite=overwrite)

# %%
if report:
    # make path
    FPATH_REPORT = os.path.join(
        FPATH_DERIVATIVES,
        'preprocessing',
        'sub-%s' % f'{subject:03}',
        'report')
    # create path on the fly
    if not Path(FPATH_REPORT).exists():
        Path(FPATH_REPORT).mkdir(parents=True, exist_ok=True)

    # create data report
    bidsdata_report = Report(title='Subject %s' % f'{subject:03}')
    bidsdata_report.add_raw(raw=raw_bl, title='Raw data',
                            butterfly=False,
                            replace=True,
                            psd=True)

    # add bad channels
    bads_html = """
    <p>Bad channels identified by PyPrep:</p>
    <p>%s</p> 
    """ % '<br> '.join([key + ' ' + str(val)
                        for key, val in bad_channels.items()])
    bidsdata_report.add_html(title='Bad channels',
                             tags='bads',
                             html=bads_html,
                             replace=True)
    # add ica
    fig = ica.plot_components(show=False, picks=np.arange(ica.n_components_))
    plt.close('all')

    bidsdata_report.add_figure(
        fig=fig,
        tags='ica',
        title='ICA cleaning',
        caption='Bad components identified by ICA Label: %s' % ', '.join(
            str(ix) + ': ' + labels[ix] for ix in exclude_idx),
        image_format='PNG',
        replace=True
    )

    for rep_ext in ['hdf5', 'html']:
        FPATH_REPORT_O = os.path.join(
            FPATH_REPORT,
            'sub-%s_task-%s_preprocessing_report.%s'
            % (f'{subject:03}', 'dpx', rep_ext))

        if overwrite:
            logger.info("`overwrite` is set to ``True`` ")

            bidsdata_report.save(FPATH_REPORT_O,
                                 overwrite=overwrite,
                                 open_browser=False)
