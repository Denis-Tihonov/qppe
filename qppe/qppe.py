import warnings
import math as m
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.signal import find_peaks
from scipy.spatial import distance_matrix
from scipy.linalg import hankel
from statsmodels.tsa.stattools import acf

############################################################################################################
class NWregression():

    def __init__(self, h, kernel = 'gaussian', metric = 'l2'):
        kernels = {
            'gaussian': self._gaussian_kernel,
            'rectangular': self._rectangular_kernel,
            'triangular': self._triangular_kernel,
            'quadratic': self._quadratic_kernel,
            'quartic': self._quartic_kernel
        }
        self.metric = metric
        self.h = h
        self.kernel = kernels[kernel]
        
    def _gaussian_kernel(self, r):
        return np.exp(-2*r**2)
    
    def _rectangular_kernel(self, r):
        return r * (r < 1)

    def _triangular_kernel(self, r):
        return (1-r) * (r < 1)

    def _quadratic_kernel(self, r):
        return (1-r**2) * (r < 1)

    def _quartic_kernel(self, r):
        return (1-r**2)**2 * (r < 1)
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        return self

    def predict(self, X):
        
        dist_matrix = pairwise_distances(X,self.X, metric = self.metric)
        
        weight = self.kernel(dist_matrix/self.h)
        
        norm_coef = np.sum(weight,axis=1).reshape((len(X),1))
        
        regression_ans = (weight@self.Y)/norm_coef
        
        return regression_ans
############################################################################################################
def delay_embedding_matrix(s, nlags):
    """Make a matrix with delay embeddings.

    Parameters
    ----------
    s : np.array
        The time series data.

    nlags : int
        Size of time lags.

    Returns
    -------
    delay_embedding_matrix : np.array of shape  (len(s) - lags + 1 , lags)
        Matrix with lags.
    """ 
    N = len(s)
    delay_embedding_matrix = hankel(s[ : N - nlags + 1], s[N - nlags : N])
    return delay_embedding_matrix
    
############################################################################################################
def _autocorr_first_period(s, nlags):
    """Find peaks of acf function to identify first period.

    Parameters
    ----------
    s : np.array
        Training time series.

    nlags : int
        Size of time lags.

    Returns
    -------
    self : np.array of shape  (len(s) - lags + 1 , lags)
        Matrix with lags.
    """ 
    acf_result = acf(
        x = s[nlags-1:],
        nlags = len(s),
        fft = True
    )
    
    peaks_indices = find_peaks(acf_result)[0]
    peaks_height = acf_result[peaks_indices]
    peaks_indices_sored_hight = sorted(range(len(peaks_indices)), key=lambda i: peaks_height[i], reverse = True)
    
    index_period_start, index_period_end = 0, peaks_indices[peaks_indices_sored_hight[0]]
    return index_period_start, index_period_end
############################################################################################################
def ro_cos(x):
    return (1 - np.cos(x))/2
    
def _loss_1(previous_phi, phi):
    assert isinstance(previous_phi, float)
    loss1 = ro_cos(np.maximum(0, previous_phi - np.array(phi)))
    return loss1

def _loss_2(near_phis, phi):
    loss2 = np.sum((1 - np.cos(near_phis[:, None] - np.array(phi)[None, :])) / 2, axis=0)
    return loss2

def _loss_3(x, x_neigh, normalization):
    x_array = np.full_like(x_neigh, x)
    loss3 = np.linalg.norm(x_array - x_neigh, axis=1) / normalization.squeeze()
    return loss3
    
############################################################################################################
def QPPE(
    x,
    nlags = 500,
    epsilon = None,
    min_dim = 4,
    l1 = 1.0,
    l2 = 1.0,
    l3 = 1.0,
    update_coef = 1.5,
    make_plots = False,
    return_variance_model = False,
    return_expectation_model = False
):

    """
    TODO: DESCRIPTION
    Parameters
    ----------
    x : np.array
        Training time series.
        
    nlags: int 
        Size of time lags.
        
    epsilon: float 
        область близости в фазовом пространстве для значения фаз
        
    min_dim  - int размерность подпространства
    
    l1 - float коэфициенты 1 лосс функции
    
    l2 - float коэфициенты 2 лосс функции
    
    l3 - float коэфициенты 3 лосс функции
    
    update_coef - float коэфициент обновления (update_coef*длинна_первого_периода_по_автокор_функции)
    
    return_variance_model - bool возвращать ли np.array модели дисперсии
    
    return_expectation_model - bool возвращать ли np.array модели математического ожидания

    Returns
    -------
    self : np.array of shape  (len(s) - lags + 1 , lags)
        Matrix with lags.
    """
    warnings.warn("The method is not optimized. Time complexity O(n^3)")

    # Autocorr and finding indeces of first period
    index_period_start, index_period_end = _autocorr_first_period(x, nlags)
    
    # PCA from initial phase space to lower dimention
    data_init = delay_embedding_matrix(x, nlags)
    X = PCA(n_components = min_dim).fit_transform(data_init)

    # Making a model in phase space
    metric = lambda x,y: (2 - 2*np.cos(x-y))**0.5
    
    phase = np.linspace(0, 2 * np.pi, index_period_end).reshape((index_period_end, 1))
    delta_phase = float(phase[1] - phase[0])

    model_expectation = NWregression(h = delta_phase,
                               metric = metric)
    model_expectation.fit(phase, X[index_period_start: index_period_end])
    expectation_array = model_expectation.predict(phase)

    variance_init = 0.25 * max(distance_matrix(expectation_array, expectation_array, p=2).reshape(-1,))
    variance = NWregression(
        h = delta_phase,
        metric = metric
    )
    variance.fit(phase, np.full((len(expectation_array), 1), variance_init))
    variance_array = variance.predict(phase)

    # Area for history point
    if epsilon is None:
        epsilon = 0.5 * variance_init

    # Implementing of phase retrieval algo 
    history_phase = []
    history_x = []
    n_points = len(expectation_array)
    indeces_array = np.arange(n_points)

    model_phases = np.linspace(0, 2 * np.pi, n_points)
    
    # Updating models parametres
    prev_update = 0
    
    for i in tqdm(np.arange(len(X))):
        # Nearest neigh at the beggining of alg
        if len(history_phase) == 0:
            model_indeces = np.argmin(distance_matrix(np.array([X[i]]), expectation_array, p=2))
            current_phi = model_phases[model_indeces]
            history_x.append(X[0])
            history_phase.append(float(current_phi))
            continue

        # Nearest neigh at the approximation function        
        model_indeces = indeces_array[(np.linalg.norm(X[i][None, :] - expectation_array, axis=1) <= variance_array.squeeze())]
        if len(model_indeces) == 0:
            model_indeces = np.argmin(distance_matrix(np.array([X[i]]), expectation_array, p = 2))
            possible_phi = np.array([model_phases[model_indeces]])
            variance_value = np.array([variance_array[model_indeces]])
            expectation_value = np.array([expectation_array[model_indeces]])
        else:
            possible_phi = model_phases[model_indeces]
            variance_value = variance_array[model_indeces]
            expectation_value = expectation_array[model_indeces]

        # Nearest neigh at the history
        near_from_history = np.array(history_phase)[(np.linalg.norm(X[i][None, :] - np.array(history_x), axis=1) <= epsilon)]
        
        # Choosing phi acording to loss function
        idx_min = np.argmin(
            l1 * _loss_1(history_phase[-1], possible_phi)
            + l2 * _loss_2(near_from_history, possible_phi)
            + l3 * _loss_3(X[i], expectation_value, variance_value)
        )

        current_phi = possible_phi[idx_min]

        # Filling in history
        history_x.append(X[i])
        history_phase.append(float(current_phi))
        
        # Updates models
        if np.abs(history_phase[-1] -  2 * np.pi) < 2*delta_phase and np.abs(i - prev_update) > update_coef * nlags:
            np_history_x = np.array(history_x)[prev_update:]
            np_history_phase = np.array(history_phase)[prev_update: , None]
            
            expectation = NWregression(h=delta_phase, metric=metric)
            variance = NWregression(h=delta_phase, metric=metric)
    
            expectation.fit(np_history_phase, np_history_x)
            
            expectation_array = expectation.predict(phase)

            variance_current = np.linalg.norm(np_history_x - expectation.predict(np_history_phase), axis=1)[:, None]
            variance.fit(np_history_phase, variance_current)
            
            variance_array = 3 * variance.predict(phase)
            prev_update = i.copy()

    # Preparing results
    result_dict = {}
    result_dict['phase'] = np.array(history_phase)

    if return_variance_model:
        result_dict['variance'] = variance_array

    if return_expectation_model:
        result_dict['expectation'] = expectation_array
    return result_dict