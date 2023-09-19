# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#          Rhys Hobbs <iskandarrhobbs@gmail.com>
#
# License: BSD-3-Clause
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np

from scipy.linalg import pinv
from scipy.stats import t

from mne.utils import logger
from mne.stats.cluster_level import _setup_adjacency, _find_clusters


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class LinearModel:
    """
    Simple linear regression model using the least squares method.

    This class allows fitting a linear model to the data and making predictions.
    It can operate with or without intercept.

    Attributes
    ----------
    coef_ : array_like or None
        Coefficients of the model. None until the model is fitted.
    fit_intercept : bool, default=False
        Whether to fit an intercept term.

    Methods
    -------
    fit(targets, predictors) :
        Fit the model to targets and predictors.
    predict(predictors) :
        Predict using the linear model.

    Properties
    ----------
    is_fitted : bool
        Returns True if the model has been fitted, else False.
    """

    def __init__(self, fit_intercept=False):
        """
        Initialize the LinearModel.

        Parameters
        ----------
        fit_intercept : bool, default=False
            Whether to fit an intercept term.
        """
        self.coef_ = None
        self.fit_intercept = fit_intercept

    @staticmethod
    def _least_squares_fit(design_matrix, y):
        """
        Compute the coefficients using the least squares method.

        Parameters
        ----------
        design_matrix : array_like
            The design matrix (predictors).
        y : array_like
            Target values.

        Returns
        -------
        array_like
            Coefficients.
        """
        return np.dot(pinv(design_matrix), y)

    @property
    def is_fitted(self):
        """
        Check if the model has been fitted.

        Returns
        -------
        bool
            True if the model has been fitted, else False.
        """
        return self.coef_ is not None

    def fit(self, targets, predictors):
        """
        Fit the linear model.

        Parameters
        ----------
        targets : array_like
            Target values.
        predictors : array_like
            Predictor values. If not a numpy array, it will be converted.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(predictors, np.ndarray):
            predictors = predictors.to_numpy()

        # compute beta coefficients
        self.coef_ = self._least_squares_fit(predictors, targets)

        return self

    def predict(self, predictors):
        """
        Predict using the linear model.

        Parameters
        ----------
        predictors : array_like
            Predictor values.

        Returns
        -------
        array_like
            Predicted values.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError('This model has not been fitted yet!')

        return np.dot(predictors, self.coef_)


def _paired_samples_ttest(data_one, data_two):
    """
    Perform a paired samples t-test.

    Parameters
    ----------
    data_one, data_two : array_like
        Two datasets for which the t-test should be computed.

    Returns
    -------
    array_like
        t-values for the test.

    Raises
    ------
    ValueError
        If the two datasets have different shapes.
    """

    if data_one.shape != data_two.shape:
        raise ValueError('\nData must be of equal dimensions!\n')

    # compute difference
    diff = data_one - data_two

    # compute t-values
    n = diff.shape[0]
    m = diff.mean(axis=0)
    sd = np.std(diff, axis=0)
    t_vals = m / (sd / np.sqrt(n))

    return t_vals


def _bootstrap(data_one, data_two, multcomp=False, random_state=None,
               adjacency=None, partitions=None):

    times = int(data_one.shape[-1]/64)

    if not isinstance(random_state, int):
        random_state = None

    #  create bootstrap sample
    # extract random subjects from overall sample
    random = np.random.RandomState(random_state)
    boot_samples = random.choice(
        range(data_one.shape[0]),
        data_one.shape[0],
        replace=True
    )

    # resampled betas
    resampled_one = data_one[boot_samples, :]
    resampled_two = data_two[boot_samples, :]

    t_vals = _paired_samples_ttest(resampled_one, resampled_two)

    if multcomp == 'fmax':

        # get maximum test statistic
        f_max = np.max(t_vals ** 2)

        return f_max

    else:
        # return t-values
        return t_vals

def bootstrap_ttest(data_one, data_two, one_sample=False,
                    nboot=1000, random=True, jobs=1,
                    multcomp=False, adjacency=None):

    if multcomp == 'fmax':
        data_one = data_one - data_one.mean(axis=0)
        if not one_sample:
            data_two = data_two - data_two.mean(axis=0)

    if random:
        random_int = np.empty(nboot)
    else:
        random_int = np.arange(nboot)

    # create parallel functions
    delayed_funcs = [
        delayed(_bootstrap)(data_one, data_two,
                            multcomp=multcomp,
                            random_state=random_int[i])
        for i in range(nboot)
    ]

    parallel_pool = ProgressParallel(n_jobs=jobs)
    out = np.array(parallel_pool(delayed_funcs))
    logger.info('\nFinished bootstrapping!\n')

    return out


class ModelInference:
    """
    Class for statistical inference on models.

    Attributes
    ----------
    t_vals_ : array_like or None
        t-values from the most recent t-test. None until a test is performed.

    Methods
    -------
    paired_ttest(data_one, data_two) :
        Perform a paired samples t-test on two datasets.
    """

    def __init__(self):
        """
        Initialize the ModelInference instance.
        """
        self.t_vals_ = None
        self.bootstrap_h0 = None

    def paired_ttest(self, data_one, data_two, adjacency=None):
        """
        Compute the t-values for a paired t-test on two datasets.

        Parameters
        ----------
        data_one, data_two : array_like
            Two datasets to compare.
        adjacency : ? (optional)
            Adjacency information for spatial clustering.
            (Currently not used, reserved for future implementations.)

        Returns
        -------
        None
        """
        self.t_vals_ = _paired_samples_ttest(data_one, data_two)


def within_subject_cis(data, ci=0.99):
    # see Morey (2008): Confidence Intervals from Normalized Data:
    # A correction to Cousineau (2005)

    # check that objects have same shape
    shapes = [x.shape for x in data]
    if len(np.unique(shapes)) > 3:
        ValueError('\nCurrently, only objects of same shape are supported.\n')

    # subjects
    subjs = np.arange(0, data[0].shape[0])
    # number of channels and time samples
    n_channels = data[0].shape[1]
    n_times = data[0].shape[2]

    # correction factor for number of conditions
    n_cond = len(data)
    corr_factor = np.sqrt(n_cond / (n_cond - 1))

    # compute condition grand averages
    grand_averages = [cond.mean(axis=0) for cond in data]

    # compute normed ERPs:
    # ((condition ERP - subject ERP) + grand average) * corr_factor
    norm_erps = np.zeros((n_cond, len(subjs), n_channels, n_times))
    for subj in subjs:
        for ncond, cond in enumerate(data):
            subj_erp = np.mean([data[0][subj, ...], data[1][subj, ...]], axis=0)
            erp_data = cond[subj, ...].copy() - subj_erp
            erp_data = (erp_data + grand_averages[ncond].data)

            norm_erps[ncond, subj, :] = erp_data

    confint = np.zeros((n_cond, n_channels, n_times))
    for n_c in range(n_cond):
        se = np.std(norm_erps[n_c, :], axis=0) / np.sqrt(len(subjs))
        confint[n_c, :] = se * t.ppf((1 + ci) / 2.0, len(subjs) - 1)

    # apply the correction factor
    confint = confint * corr_factor

    return confint