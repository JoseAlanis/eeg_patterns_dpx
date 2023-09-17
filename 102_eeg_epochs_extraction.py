"""
==================================
Extract EEG segments from the data
==================================

Segment EEG data around experimental events

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
import matplotlib.patches as patches

from mne import events_from_annotations, Epochs, open_report
from mne.io import read_raw_fif
from mne.utils import logger

from autoreject import AutoReject
from autoreject import set_matplotlib_defaults

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
gat = False

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        overwrite=overwrite,
        report=report,
        jobs=jobs,
        gat=gat
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    overwrite = defaults["overwrite"]
    report = defaults["report"]
    jobs = defaults["jobs"]
    gat = defaults["gat"]

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
# create bids path for import
raw_fname = os.path.join(
    FPATH_DERIVATIVES,
    'preprocessing',
    'sub-%s' % f'{subject:03}',
    'eeg',
    'sub-%s_task-%s_preprocessed-raw.fif' % (f'{subject:03}', 'dpx'))
# get the data
raw = read_raw_fif(raw_fname, preload=True)

# %%
# only keep EEG channels
raw.pick(['eeg'])

# %%
events, event_ids = events_from_annotations(raw, regexp=None)

# get the correct trigger channel values for each event category
cue_vals = []
for key, value in event_ids.items():
    if key.startswith('cue'):
        cue_vals.append(value)

cue_b_vals = []
for key, value in event_ids.items():
    if key.startswith('cue_b'):
        cue_b_vals.append(value)

probe_vals = []
for key, value in event_ids.items():
    if key.startswith('probe'):
        probe_vals.append(value)

probe_y_vals = []
for key, value in event_ids.items():
    if key.startswith('probe_y'):
        probe_y_vals.append(value)

correct_reactions = []
for key, value in event_ids.items():
    if key.startswith('correct'):
        correct_reactions.append(value)

incorrect_reactions = []
for key, value in event_ids.items():
    if key.startswith('incorrect'):
        incorrect_reactions.append(value)

# %%
# global variables
trial = 0
broken = []
sfreq = raw.info['sfreq']
block_end = events[events[:, 2] == event_ids['EDGE boundary'], 0] / sfreq

# placeholders for results
block = []
probe_ids = []
reaction = []
rt = []

# copy of events
new_evs = events.copy()

# loop trough events and recode them
for event in range(len(new_evs[:, 2])):

    # *** if event is a cue stimulus ***
    if new_evs[event, 2] in cue_vals:

        # save block based on onset (before or after break)
        if (new_evs[event, 0] / sfreq) < block_end:
            block.append(0)
        else:
            block.append(1)

        # --- 1st check: if next event is a false reaction ---
        if new_evs[event + 1, 2] in incorrect_reactions:
            # if event is an A-cue
            if new_evs[event, 2] == event_ids['cue_a']:
                # recode as too soon A-cue
                new_evs[event, 2] = 118
            # if event is a B-cue
            elif new_evs[event, 2] in cue_b_vals:
                # recode as too soon B-cue
                new_evs[event, 2] = 119

            # look for next probe
            i = 2
            while new_evs[event + i, 2] not in probe_vals:
                if new_evs[event + i, 2] in cue_vals:
                    broken.append(trial)
                    break
                i += 1

            # if probe is an X
            if new_evs[event + i, 2] == event_ids['probe_x']:
                # recode as too soon X-probe
                new_evs[event + i, 2] = 120
            # if probe is an Y
            elif new_evs[event + i, 2] in probe_y_vals:
                # recode as too soon Y-probe
                new_evs[event + i, 2] = 121

            # save trial information as NaN
            trial += 1
            rt.append(np.nan)
            reaction.append(np.nan)
            # go on to next trial
            continue

        # *** 2nd check: if next event is a probe stimulus ***
        elif new_evs[event + 1, 2] in probe_vals:

            # if event after probe is a reaction
            if new_evs[event + 2, 2] in correct_reactions + incorrect_reactions:

                # save reaction time
                rt.append(
                    (new_evs[event + 2, 0] - new_evs[event + 1, 0]) / sfreq)

                # if reaction is correct
                if new_evs[event + 2, 2] in correct_reactions:

                    # save response
                    reaction.append(1)

                    # if cue was an A
                    if new_evs[event, 2] == event_ids['cue_a']:
                        # recode as correct A-cue
                        new_evs[event, 2] = 122

                        # if probe was an X
                        if new_evs[event + 1, 2] == event_ids['probe_x']:
                            # recode as correct AX probe combination
                            new_evs[event + 1, 2] = 123

                        # if probe was a Y
                        else:
                            # recode as correct AY probe combination
                            new_evs[event + 1, 2] = 124

                        # go on to next trial
                        trial += 1
                        continue

                    # if cue was a B
                    else:
                        # recode as correct B-cue
                        new_evs[event, 2] = 125

                        # if probe was an X
                        if new_evs[event + 1, 2] == event_ids['probe_x']:
                            # recode as correct BX probe combination
                            new_evs[event + 1, 2] = 126
                        # if probe was a Y
                        else:
                            # recode as correct BY probe combination
                            new_evs[event + 1, 2] = 127

                        # go on to next trial
                        trial += 1
                        continue

                # if reaction was incorrect
                elif new_evs[event + 2, 2] in incorrect_reactions:

                    # save response
                    reaction.append(0)

                    # if cue was an A
                    if new_evs[event, 2] == event_ids['cue_a']:
                        # recode as incorrect A-cue
                        new_evs[event, 2] = 128

                        # if probe was an X
                        if new_evs[event + 1, 2] == event_ids['probe_x']:
                            # recode as incorrect AX probe combination
                            new_evs[event + 1, 2] = 129

                        # if probe was a Y
                        else:
                            # recode as incorrect AY probe combination
                            new_evs[event + 1, 2] = 130

                        # go on to next trial
                        trial += 1
                        continue

                    # if cue was a B
                    else:
                        # recode as incorrect B-cue
                        new_evs[event, 2] = 131

                        # if probe was an X
                        if new_evs[event + 1, 2] == event_ids['probe_x']:
                            # recode as incorrect BX probe combination
                            new_evs[event + 1, 2] = 132

                        # if probe was a Y
                        else:
                            # recode as incorrect BY probe combination
                            new_evs[event + 1, 2] = 133

                        # go on to next trial
                        trial += 1
                        continue

            # if no reaction followed cue-probe combination
            elif new_evs[event + 2, 2] not in \
                    correct_reactions + correct_reactions:

                # save reaction time as NaN
                rt.append(99999)
                reaction.append(np.nan)

                # if cue was an A
                if new_evs[event, 2] == event_ids['cue_a']:
                    # recode as missed A-cue
                    new_evs[event, 2] = 134

                    # if probe was an X
                    if new_evs[event + 1, 2] == event_ids['probe_x']:
                        # recode as missed AX probe combination
                        new_evs[event + 1, 2] = 135

                    # if probe was a Y
                    else:
                        # recode as missed AY probe combination
                        new_evs[event + 1, 2] = 136

                    # go on to next trial
                    trial += 1
                    continue

                # if cue was a B
                else:
                    # recode as missed B-cue
                    new_evs[event, 2] = 137

                    # if probe was an X
                    if new_evs[event + 1, 2] == event_ids['probe_x']:
                        # recode as missed BX probe combination
                        new_evs[event + 1, 2] = 138

                    # if probe was a Y
                    else:
                        # recode as missed BY probe combination
                        new_evs[event + 1, 2] = 139

                    # go on to next trial
                    trial += 1
                    continue

    # skip other events
    else:
        continue


# %%
# cue events
cue_event_id = {'Too_soon A': 118,
                'Too_soon B': 119,

                'Correct A': 122,
                'Correct B': 125,

                'Incorrect A': 128,
                'Incorrect B': 131,

                'Missed A': 134,
                'Missed B': 137}

# probe events
probe_event_id = {'Too_soon X': 120,
                  'Too_soon Y': 121,

                  'Correct AX': 123,
                  'Correct AY': 124,

                  'Correct BX': 126,
                  'Correct BY': 127,

                  'Incorrect AX': 129,
                  'Incorrect AY': 130,

                  'Incorrect BX': 132,
                  'Incorrect BY': 133,

                  'Missed AX': 135,
                  'Missed AY': 136,

                  'Missed BX': 138,
                  'Missed BY': 139}


# %%
# only keep cue events
cue_events = new_evs[np.where((new_evs[:, 2] == 118) |
                              (new_evs[:, 2] == 119) |
                              (new_evs[:, 2] == 122) |
                              (new_evs[:, 2] == 125) |
                              (new_evs[:, 2] == 128) |
                              (new_evs[:, 2] == 131) |
                              (new_evs[:, 2] == 134) |
                              (new_evs[:, 2] == 137))]

# only keep probe events
probe_events = new_evs[np.where((new_evs[:, 2] == 120) |
                                (new_evs[:, 2] == 121) |
                                (new_evs[:, 2] == 123) |
                                (new_evs[:, 2] == 124) |
                                (new_evs[:, 2] == 126) |
                                (new_evs[:, 2] == 127) |
                                (new_evs[:, 2] == 129) |
                                (new_evs[:, 2] == 130) |
                                (new_evs[:, 2] == 132) |
                                (new_evs[:, 2] == 133) |
                                (new_evs[:, 2] == 135) |
                                (new_evs[:, 2] == 136) |
                                (new_evs[:, 2] == 138) |
                                (new_evs[:, 2] == 139))]


# %%
# reversed event_id dict
cue_event_id_rev = {val: key for key, val in cue_event_id.items()}
probe_event_id_rev = {val: key for key, val in probe_event_id.items()}

# make sure events shapes match
if cue_events.shape[0] != probe_events.shape[0]:
    cue_events = np.delete(cue_events, broken, 0)

# %%
# create list with reactions based on cue and probe event ids
same_stim, reaction_cues, reaction_probes, cues, probes, reaction = \
    [], [], [], [], [], []

for cue, probe in zip(cue_events[:, 2], probe_events[:, 2]):
    response, cue = cue_event_id_rev[cue].split(' ')
    reaction_cues.append(response)
    # save cue
    cues.append(cue)

    # save response
    response, probe = probe_event_id_rev[probe].split(' ')
    reaction_probes.append(response)

    if response == 'Correct':
        reaction.append(probe)
    elif response == 'Incorrect':
        if probe == 'AX' and response == 'Incorrect':
            reaction.append('AY')
        elif probe in ['BX', 'BY', 'AY'] and response == 'Incorrect':
            reaction.append('AX')
    else:
        reaction.append(np.nan)

    # check if same type of combination was shown in the previous trail
    if len(probes):
        stim = same_stim[-1]
        if probe == probes[-1] \
                and response == 'Correct' \
                and reaction_probes[-2] == 'Correct':
            stim += 1
            same_stim.append(stim)
        else:
            same_stim.append(0)
    else:
        stim = 0
        same_stim.append(0)

    # save probe
    probes.append(probe)

# %%
# create data frame with epochs metadata
metadata = {'block': np.delete(block, broken, 0),
            'trial': np.delete(np.arange(0, trial), broken, 0),
            'cue': cues,
            'probe': [probe[-1] for probe in probes],
            'cue_probe_combination': probes,
            'run': same_stim,
            'reaction_cue': reaction_cues,
            'reaction_probe': reaction_probes,
            'reaction_combination': reaction,
            'rt': np.delete(rt, broken, 0)}
metadata = pd.DataFrame(metadata)

# %%
# save RT measures for later analyses
rt_data = metadata.copy()
rt_data = rt_data.assign(subject=subject)

# %%
# condition and analysis specific parameters

# lower the sampling rate
decim = 1
if raw.info['sfreq'] == 256.0:
    decim = 2
elif raw.info['sfreq'] == 512.0:
    decim = 4
elif raw.info['sfreq'] == 1024.0:
    decim = 8

# conditions dictionary
conditions = {'cues': [cue_events, cue_event_id],
              'probes': [probe_events, probe_event_id]}

# initialize threshold figure
fig, ax = plt.subplots(nrows=len(conditions),
                       ncols=1,
                       figsize=(15, 10))

# initialize autoreject figure
fig_log, axl = plt.subplots(nrows=len(conditions),
                            ncols=1,
                            figsize=(15, 10))

# labels
if gat:
    gat_label = ' (for GAT)'
else:
    gat_label = ''

# %%
# loop through conditions and extract epochs
for condition in conditions:

    cond_id = list(conditions.keys()).index(condition)

    # extract condition specific epochs
    evs, ids = conditions[condition]
    if condition == 'cues':
        if gat:
            times = [-0.5, 3.5]
        else:
            times = [-0.5, 2.5]
    else:
        times = [-0.5, 1.0]
    epochs = Epochs(raw, evs, ids,
                    metadata=metadata,
                    on_missing='ignore',
                    tmin=times[0],
                    tmax=times[1],
                    baseline=None,
                    preload=True,
                    reject_by_annotation=True,
                    decim=decim
                    )

    # use autoreject algorithm to find and repair bad epochs
    ar = AutoReject(n_jobs=jobs, random_state=42)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # save reaction variable in rt dataframe
    if condition == 'cues':
        rt_data = rt_data.assign(rejected_cues=reject_log.bad_epochs)
    else:
        rt_data = rt_data.assign(rejected_probes=reject_log.bad_epochs)

    # plot autoreject results
    set_matplotlib_defaults(plt, style='seaborn-white')

    # autoreject settings
    consensus = np.array(ar.consensus)
    n_interpolate = np.array(ar.n_interpolate)
    error = ar.loss_['eeg'].mean(axis=-1).copy()

    im = ax[cond_id].imshow(error.T * 1e6, cmap=plt.get_cmap('viridis'))

    ax[cond_id].set_xticks(range(len(consensus)),
                           ['%.1f' % c for c in consensus])
    ax[cond_id].set_yticks(range(len(n_interpolate)), n_interpolate)

    # draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(error.argmin(), error.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax[cond_id].add_patch(rect)
    ax[cond_id].xaxis.set_ticks_position('bottom')

    ax[cond_id].set_xlabel(r'Consensus percentage $\kappa$%s' % gat_label)
    ax[cond_id].set_ylabel(r'Max sensors interpolated $\rho$%s' % gat_label)
    ax[cond_id].set_title('Mean cross validation error (x 1e6), %s%s'
                          % (condition, gat_label))
    cax = ax[cond_id].inset_axes([1.04, 0.2, 0.03, 0.6])
    fig.colorbar(im, cax=cax)
    plt.close('all')

    # plot epochs statistics
    axl[cond_id].set_title('Interpolated epochs, %s%s'
                           % (condition, gat_label))
    reject_log.plot(orientation="horizontal", ax=axl[cond_id], show=False)
    plt.close('all')

    # create path for preprocessed dara
    FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                                'gat/epochs' if gat else 'epochs',
                                'sub-%s' % f'{subject:03}',
                                'sub-%s_%s-epo.fif'
                                % (f'{subject:03}', condition))

    FPATH_AR = os.path.join(FPATH_DERIVATIVES,
                            'gat/epochs' if gat else 'epochs',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_%s_reject-log.npz'
                            % (f'{subject:03}', condition))

    # check if directory exists
    if not Path(FPATH_EPOCHS).exists():
        Path(FPATH_EPOCHS).parent.mkdir(parents=True, exist_ok=True)

    # save cue epochs to disk
    epochs_clean.save(FPATH_EPOCHS, overwrite=overwrite)
    reject_log.save(FPATH_AR, overwrite=overwrite)

# %%
# write report

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

    bad_epochs_cues = rt_data.loc[rt_data['rejected_cues']].trial
    bad_epochs_probes = rt_data.loc[rt_data['rejected_probes']].trial

    bads_html = """
    <p>Bad epochs identified by Autoreject%s:</p>
    <p>Cues:</p> 
    <p>%s</p> 
    <p>Probes:</p> 
    <p>%s</p> 
    """ % (gat_label,
           ', '.join(str(bad) for bad in bad_epochs_cues),
           ', '.join(str(bad) for bad in bad_epochs_probes))

    bidsdata_report.add_html(title='Bad epochs%s' % gat_label,
                             tags='epochs',
                             html=bads_html,
                             replace=True)

    bidsdata_report.add_figure(
        fig=fig,
        tags='epochs',
        title='Autoreject parameters%s' % gat_label,
        image_format='PNG',
        replace=True
    )

    bidsdata_report.add_figure(
        fig=fig_log,
        tags='epochs',
        title='Autoreject rejection log%s' % gat_label,
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

# %%
# save RT data
if not gat:
    # create path for rt data
    FPATH_RT = os.path.join(FPATH_DERIVATIVES,
                            'rt',
                            'sub-%s' % f'{subject:03}',
                            'sub-%s_rt.tsv' % f'{subject:03}')

    # check if directory exists
    if not Path(FPATH_RT).exists():
        Path(FPATH_RT).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(FPATH_RT) and not overwrite:
        raise RuntimeError(
            f"'{FPATH_RT}' already exists; consider setting 'overwrite' to True"
        )

    # save rt data to disk
    rt_data.to_csv(FPATH_RT,
                   sep='\t',
                   index=False)
