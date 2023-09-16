"""
===========================
Source data set to EEG BIDS
===========================

Put EEG data into a BIDS-compliant directory structure.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os

from pathlib import Path

import pandas as pd

from mne import find_events, Report
from mne.io import read_raw_bdf
from mne.utils import logger

from mne_bids import BIDSPath, write_raw_bids

from config import (
    FPATH_SOURCEDATA,
    FPATH_BIDS,
    FPATH_DERIVATIVES,
    MISSING_FPATH_SOURCEDATA_MSG,
    FNAME_SOURCEDATA_TEMPLATE,
    SUBJECT_IDS,
    montage,
    event_id,
    anonym
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
report = False
overwrite = False
ext = '.bdf'

# %%
# When not in an IPython, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        report=report,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    report = defaults["report"]
    overwrite = defaults["overwrite"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(f"{subject} is not a valid subject ID.\n"
                     f"Use one of: {SUBJECT_IDS}")

if not os.path.exists(FPATH_SOURCEDATA):
    raise RuntimeError(
        MISSING_FPATH_SOURCEDATA_MSG.format(FPATH_SOURCEDATA)
    )
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
# path to file in question (i.e.,  subject to analyse)
fname = FNAME_SOURCEDATA_TEMPLATE.format(
    subject=subject,
    dtype='eeg',
    ext=ext
)

# %%
# 1) import the data
raw = read_raw_bdf(fname,
                   preload=False)
# get sampling frequency
sfreq = raw.info['sfreq']
# channels names
channels = raw.info['ch_names']

# identify channel types based on matching names in montage
types = []
for channel in channels:
    if channel in montage.ch_names:
        types.append('eeg')
    elif channel.startswith('EOG') | channel.startswith('EXG'):
        types.append('eog')
    else:
        types.append('stim')

# add channel types and eeg-montage
types = {channel: typ for channel, typ in zip(channels, types)}
raw.set_channel_types(types)
raw.set_montage(montage)

# %%
# 2) add subject info

# compute approx. date of birth
# get measurement date from dataset info
date_of_record = raw.info['meas_date']
# convert to date format
date = date_of_record.strftime('%Y-%m-%d')

# here, we compute only and approximate of the subject's birthday
# this is to keep the date anonymous (at least to some degree)
demographics = FNAME_SOURCEDATA_TEMPLATE.format(subject=subject,
                                                dtype='demographics',
                                                ext='.tsv')
demo = pd.read_csv(demographics, sep='\t', header=0)
age = demo[demo.subject_id == 'sub-' + f'{subject:03}'].age[0]
sex = demo[demo.subject_id == 'sub-' + f'{subject:03}'].sex[0]

year_of_birth = int(date.split('-')[0]) - age
approx_birthday = (year_of_birth,
                   int(date[5:].split('-')[0]),
                   int(date[5:].split('-')[1]))

# add modified subject info to dataset
raw.info['subject_info'] = dict(id=subject,
                                sex=sex,
                                birthday=approx_birthday)

# frequency of power line
raw.info['line_freq'] = 50.0

# %%
# 3) get eeg events
events = find_events(raw,
                     stim_channel='Status',
                     output='onset',
                     min_duration=0.0)
# only keep relevant events
keep_evs = [events[i, 2] in event_id.values() for i in range(events.shape[0])]
events = events[keep_evs]

# %%
# 4) export to bids

# create bids path
output_path = BIDSPath(subject=f'{subject:03}',
                       task='dpx',
                       datatype='eeg',
                       root=FPATH_BIDS)

# write file
write_raw_bids(raw,
               anonymize={'daysback': anonym['daysback'],
                          'keep_his': True},
               events=events,
               event_id=event_id,
               bids_path=output_path,
               overwrite=overwrite)

# %%
if report:
    bidsdata_report = Report(title='Subject %s' % f'{subject:03}')
    bidsdata_report.add_raw(raw=raw, title='Raw data',
                            butterfly=False,
                            replace=True,
                            psd=True)

    FPATH_REPORT = os.path.join(FPATH_DERIVATIVES,
                                'report',
                                'sub-%s' % f'{subject:03}')

    if not Path(FPATH_REPORT).exists():
        Path(FPATH_REPORT).mkdir(parents=True, exist_ok=True)

    if overwrite:
        logger.info("`overwrite` is set to ``True`` ")

    for rep_ext in ['hdf5', 'html']:
        FPATH_REPORT_O = os.path.join(
            FPATH_REPORT,
            'Subj_%s_preprocessing_report.%s' % (f'{subject:03}', rep_ext))

        bidsdata_report.save(FPATH_REPORT_O,
                             overwrite=overwrite,
                             open_browser=False)
