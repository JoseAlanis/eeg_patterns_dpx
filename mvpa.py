"""
==============
MVPA functions
==============
Utility functions for estimation and statistical analysis of MVPA parameters.
Authors: Functions retrieved and adapted from
         https://github.com/heikele/GAT_n4-p6, also see:
         Heikel, E., Sassenhagen, J., & Fiebach, C. J. (2018).
         Time-generalized multivariate analysis of EEG responses reveals a
         cascading architecture of semantic mismatch processing.
         Brain and language, 184, 43-53.
         Changes made by José C. García Alanis <alanis.jcg@gmail.com>
License: BSD (3-clause)
"""
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import pandas as pd

from scipy.stats import wilcoxon, ttest_1samp

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mne.decoding import cross_val_multiscore, GeneralizingEstimator, \
    get_coef, Vectorizer
from mne.parallel import parallel_func
from mne.stats import fdr_correction, bonferroni_correction, \
    spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
from mne import read_epochs

from config import FPATH_DERIVATIVES


# signed rank test
def _my_wilcoxon(x):
    out = wilcoxon(x)
    return out[1]


# signed rank test
def _my_ttest(x):
    out = ttest_1samp(x, popmean=0.0)
    return out[1]


# loop function
def _loop(x, function):
    out = list()
    for ii in range(x.shape[1]):
        out.append(function(x[:, ii]))
    return out


# correct p values for multiple testing
def parallel_stats(x, function=_my_ttest, correction='Bonferroni', jobs=1):

    # check if correction method was provided
    if correction not in [False, None, 'FDR', 'Bonferroni']:
        raise ValueError('Unknown correction')

    # reshape to 2D
    x = np.array(x)
    dims = x.shape
    x.resize([dims[0], np.prod(dims[1:])])

    # prepare parallel
    n_cols = x.shape[1]
    parallel, pfunc, n_jobs = parallel_func(_loop, jobs)
    n_chunks = min(n_cols, n_jobs)
    chunks = np.array_split(range(n_cols), n_chunks)
    p_values = parallel(pfunc(x[:, chunk], function) for chunk in chunks)
    p_values = np.reshape(np.hstack(p_values), dims[1:])
    x.resize(dims)

    # apply correction
    if correction == 'FDR':
        dims = p_values.shape
        _, p_values = fdr_correction(p_values)
        p_values = np.reshape(p_values, dims)
    elif correction == 'Bonferroni':
        dims = p_values.shape
        _, p_values = bonferroni_correction(p_values)
        p_values = np.reshape(p_values, dims)

    return p_values


# one sample t-test
def _stat_fun(x, sigma=0, method='relative'):
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


# threshold free cluster permutation test
def stats_tfce(x, adjacency=None, n_permutations=1000, threshold=None, jobs=1):

    # calculate p-values using cluster permutation test
    _, _, p_values, _ = spatio_temporal_cluster_1samp_test(
        x,
        out_type='indices',
        stat_fun=_stat_fun,
        n_permutations=n_permutations,
        adjacency=adjacency,
        threshold=threshold,
        n_jobs=jobs)

    p_values = p_values.reshape(x.shape[1:])

    return p_values


# function to import eeg segments for each participant
def get_epochs(subj, query=None):
    """
    Loads the single trial data for a participant (name)
    """

    str_subj = str(subj).rjust(3, '0')
    fpath_epochs = os.path.join(FPATH_DERIVATIVES,
                                'gat/epochs',
                                'sub-%s' % str_subj,
                                'sub-%s_cues-epo.fif' % str_subj)

    epoch = read_epochs(fpath_epochs)
    epoch.apply_baseline((-0.300, -0.050))

    epoch.metadata['shifted_response'] = \
        epoch.metadata['reaction_probe'].shift(1)

    # remove epochs that preceded erroneous responses
    epoch = epoch['shifted_response != "Too_soon"']
    epoch = epoch['shifted_response != "Incorrect"']
    epoch = epoch['reaction_probe == "Correct"']
    epoch = epoch['rt > 0.1 & rt < 1.0']

    if query is not None:
        epoch = epoch[query]

    return epoch


# run generalisation across time and condition
def run_gat(subj, contrast, jobs, decoder="ridge"):
    """
    Function to run Generalization Across Time (GAT).
    Parameters
    ----------
    subj: int
    jobs: int
        The number of jobs to run in parallel.
    contrast
    decoder: str
        Specify type of classifier -'ridge' for Ridge Regression (default),
        'lin-svm' for linear SVM 'svm' for nonlinear (RBF) SVM and 'log_reg'
        for Logistic Regression.
    """

    epochs = get_epochs(subj, query=contrast)

    # specify whether to use a linear or nonlinear SVM if SVM is used
    lin = ''  # if not svm it doesn't matter, both log_reg and ridge are linear
    if "svm" in decoder:
        lin, decoder = decoder.split("-")

    # build classifier pipeline #
    # pick a machine learning algorithm to use (ridge/SVM/logistic regression)
    decoder_dict = {
        "ridge": RidgeClassifier(class_weight='balanced',
                                 random_state=42,
                                 solver="sag"),
        "svm": SVC(class_weight='balanced',
                   kernel=("rbf" if "non" in lin else "linear"),
                   random_state=42),
        "log_reg": LogisticRegression(class_weight='balanced',
                                      random_state=42)}

    # get data and targets
    data = epochs.get_data()
    labels = epochs.metadata.cue_probe_combination.to_numpy()

    # create classifier pipeline
    clf = make_pipeline(StandardScaler(),
                        decoder_dict[decoder])
    gen_clf = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=jobs)

    # compute cross validated performance scores
    scores = cross_val_multiscore(gen_clf, data,
                                  labels,
                                  cv=5,
                                  n_jobs=jobs)
    scores = scores.mean(axis=0)

    # calculate prediction confidence scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.empty((0, data.shape[-1], data.shape[-1]))
    for train, test in cv.split(data, labels):
        gen_clf.fit(data[train], labels[train])
        d = gen_clf.decision_function(data[test])
        preds = np.concatenate([preds, d])

    # compute topographical patterns
    dat = Vectorizer().fit_transform(data)
    clf.fit(dat, labels)
    dat = dat - dat.mean(axis=0, keepdims=True)

    # look for the type of classifier and get the weights
    if decoder == 'ridge':
        filt_ = clf.named_steps.ridgeclassifier.coef_.copy()
    elif decoder == 'svm':
        filt_ = clf.named_steps.svc.coef_.copy()
    elif decoder == 'log_reg':
        filt_ = clf.named_steps.logisticregression.coef_.copy()

    # Compute patterns using Haufe's trick: A = Cov_X . W . Precision_Y
    # cf.Haufe, et al., 2014, NeuroImage,
    # doi:10.1016/j.neuroimage.2013.10.067
    inv_y = 1.
    patt_ = np.cov(dat.T).dot(filt_.T.dot(inv_y)).T

    # store the patterns accordingly
    if decoder == 'ridge':
        clf.named_steps.ridgeclassifier.patterns_ = patt_
    elif decoder == 'svm':
        clf.named_steps.svc.patterns_ = patt_
    elif decoder == 'log_reg':
        clf.named_steps.logisticregression.patterns_ = patt_

    # back transform using steps in pipeline
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)

    # return subject scores, prediction confidence and topographical patterns
    return scores, preds, patterns


def get_p_scores(scores, chance=0.5,
                 tfce=False, adjacency=None,
                 permutations=1000, threshold=None, jobs=1):
    """
    Calculate p_values from scores for significance masking using either
    TFCE or FDR.
    Parameters
    ----------
    scores: numpy array
        Calculated scores from decoder
    chance: float
        Indicate chance level
    tfce: True | False
        Specify whether to Threshold Free Cluster Enhancement (True)
        or FDR (False)
    permutations: int
        The number of permutations to compute.
    threshold: float | dict | None
        Threshold that needs to be exceeded to achieve significance
    jobs: int
        The number of jobs to run in parallel (default 1). Requires the
        joblib package.
    """
    p_values = (parallel_stats(scores - chance, jobs=jobs) if tfce is False
                else stats_tfce(scores - chance,
                                n_permutations=permutations,
                                adjacency=adjacency,
                                threshold=threshold,
                                jobs=jobs))
    return p_values


def grouper(iterable):
    """
    List of time points of significance, identifies neighbouring time points.
    """
    prev = None
    group = []
    for item in iterable:
        if not prev or round(item - prev, 2) <= .01:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def find_clus(sig_times):
    """
    Identify time points of significance from FDR correction, results in
    lists of ranges and individual time points of significance and
    creates a dictionary for later use.
    Parameters
    ----------
    sig_times: list
        List of significant time points
    """
    group = dict(enumerate(grouper(sig_times)))
    clus = []
    for key in group.keys():
        ls = group[key]
        clus.append((([ls[0], ls[-1]] if round((ls[1] - ls[0]), 2) <= 0.01
                      else [ls[1], ls[-1]]) if len(group[key]) > 1
                     else group[key]))
    return clus


def get_stats_lines(scores, times, test_times, alphas=[0.05, 0.01],
                    correction='FDR'):
    """
    Calculate subject level decoder performances for each of the times series
    plots and perform FDR correction (p<0.05 and p<0.01).
    Creates a dictionary of relevant stats.
    Parameters
    ----------
    scores: array
    times: array
    test_times: dict
    alphas: list
    correction: str
    """

    # get alpha levels
    alpha1, alpha2 = alphas

    # get the diagonal (training_t==testing_t) for each subject,
    # FDR or Bonferroni correction, and mask time points of significance
    diag = np.asarray([sc.diagonal() for sc in scores])
    diag_pvalues = parallel_stats(list(diag - 0.5), correction=correction)
    diag1, diag2 = times[diag_pvalues < alpha1], times[diag_pvalues < alpha2]

    # get component boundaries
    min_max = {k: [(np.abs(v[0] - times)).argmin(),
                   (np.abs(v[1] - times)).argmin()]
               for (k, v) in test_times.items()}

    # average classifier performance over time for time window of interest
    # for each subject
    class_performance = {k: scores[:, v[0]:v[1], :].mean(1)
                         for (k, v) in min_max.items()}

    # p-value correction and significance testing
    p_vals = {k: parallel_stats(list(v - 0.5), correction=correction)
              for (k, v) in class_performance.items()}

    # mask time points of significance that are p<0.05 (alpha1) and
    # p<0.01 (alpha2)
    masks = {k: [times[v < alpha] for alpha in alphas]
             for (k, v) in p_vals.items()}

    # # *** keep this just in case we neeed it later ***
    # # average difference between classifier performance over time for
    # time window of interest for each subject
    # diff = np.array([(sc[p6_min:p6_max].mean(0) -
    #                   sc[n4_min:n4_max].mean(0))
    #                  for sc in scores])
    # # FDR correction and significance masking
    # diff_pvalues = parallel_stats(list(diff))
    # diff1, diff2 = xx[diff_pvalues < alpha1], xx[diff_pvalues < alpha2]

    # create dict of diagonal stats
    diag_stats = {'diag': [diag, diag1, diag2]}

    # create dict of classifier stats
    class_stats = {k: [v, m[0], m[1]]
                   for ((k, v), (a, m)) in
                   zip(class_performance.items(), masks.items())}

    # object for return
    stats_dict = {**diag_stats, **class_stats}

    return stats_dict


def get_dfs(stats_dict, tmin=None, df_type=False):
    """Create DataFrames for time series plotting"""
    from config import SUBJECT_IDS

    # get times
    times = get_epochs(SUBJECT_IDS[0]).times
    if tmin is not None:
        times = times[times >= tmin]

    if not df_type:
        # create dataframe for N400 and P600 decoders
        df = pd.DataFrame()
        sub, time, accuracy, comp = [], [], [], []
        comps = list(stats_dict.keys())
        comps = [i for i in comps if i != 'diag']
        for c in comps:
            for ii, s in enumerate(stats_dict[c][0]):
                for t, a in enumerate(s):
                    sub.append(ii)
                    accuracy.append(a)
                    time.append(times[t])
                    comp.append(c)
        df["Time (s)"] = time
        df["Subject"] = sub
        df["Accuracy (%)"] = accuracy
        df["Component"] = comp

    else:
        # create dataframe for diagonal or difference between components
        sub, time, ac = [], [], []
        df = pd.DataFrame()
        for ii, s in enumerate(stats_dict[df_type][0]):
            for t, a in enumerate(s):
                sub.append(ii), ac.append(a), time.append(times[t])
        df["Time (s)"], df["Subject"] = time, sub
        df["{}".format(("Accuracy (%)" if df_type == "diag"
                        else "Difference in Accuracy (%)"))] = ac

    return df