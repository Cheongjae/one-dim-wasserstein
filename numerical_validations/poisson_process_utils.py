import numpy as np
from numpy import random
from scipy.stats import poisson, expon
from scipy.special import comb
#from scipy import math
from numpy.random import exponential

from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson
from scipy.stats import *


def sample_inhomogeneous_poisson_process(time_window, rate_func, **kwarg):
    T = np.arange(time_window)
    Y = rate_func(T)
    
    tf = TimeFunction((T,Y), dt=0.01)
    in_poi = SimuInhomogeneousPoisson([tf], end_time=time_window, verbose=False)
    in_poi.simulate()
    
    return in_poi.timestamps[0]


def sample_poisson_process_from_exp(rate=0.5, window_size=100, start_t=0):
    
    scale = rate ** (-1)   
    points = []
    
    def add_point(points, scale):
        wait_time = exponential(scale)
        if len(points) == 0:
            last = start_t
        else:
            last = points[-1]
        points.append(last + wait_time)
    
    add_point(points, scale)
    while points[-1] < window_size:
        add_point(points, scale)
    if points[-1] >= window_size:
        points.pop(-1)
    return np.array(points)


def sample_poisson_process(rate=0.5, window_size=100):
    scale = rate ** (-1)
    number_points = np.random.poisson(window_size / scale)
    points = np.random.uniform(low=0, high=window_size, size=number_points)
    points.sort()
    return points


def spike_pair_wasserstein(x, y, n, m):
    return np.abs(x[n-1] - y[m-1])


def theoretical_mean(rate1, rate2, n, m):
    denom = rate1 * rate2 * (rate1+rate2) ** (n+m-1)
    num = m*rate1**(n+m) + n*rate2**(n+m)
    
    first = 0.0
    second = 0.0
    
    if n-1 >= 1:
        for k in range(1, n):
            first += (n * comb(n+m-1, k, exact=True) - m * comb(n+m-1, k-1, exact=True)) * rate1**k * rate2**(n+m-k)
    
    if m-1 >= 1:
        for k in range(1, m):
            second += (m * comb(n+m-1, k, exact=True) - n * comb(n+m-1, k-1, exact=True)) * rate1**(n+m-k) * rate2**k
    
    num += first + second
    
    return num / denom


def theoretical_mean_v2(rate1, rate2, n, m):
    denom = rate1 * rate2 * (rate1+rate2) ** (n+m-1)
    num = 0 #m*rate1**(n+m) + n*rate2**(n+m)
    
    first = 0.0
    second = 0.0
    
    for k in range(n):
        first += ((n-k) * comb(n+m, k, exact=True)) * rate1**k * rate2**(n+m-k)
    
    for k in range(m):
        second += ((m-k) * comb(n+m, k, exact=True)) * rate1**(n+m-k) * rate2**k
    
    num += first + second
    
    return num / denom


def theoretical_mean_v3(rate1, rate2, n, m):
    denom = rate1 * rate2 * (rate1+rate2) ** (n+m-1)
    num = - (rate1**(n+m) + rate2**(n+m))
    
    first = 0.0
    second = 0.0
    

    for k in range(m):
        first += k * comb(n+m, n+k, exact=True) * rate1**(n+k) * rate2**(m-k) 
    
    for k in range(n):
        second += k * comb(n+m, m+k, exact=True) * rate1**(n-k) * rate2**(m+k) 
    
    num += first + second
    
    return num / denom



def theoretical_mean_revisit(rate1, rate2, n, m):
    denom = rate1 * rate2 * (rate1+rate2) ** (n+m-2)
    num = m*rate1**(n+m) + n*rate2**(n+m)
    
    first = 0.0
    second = 0.0
    
    if n-1 >= 1:
        for k in range(1, n):
            first += (n * comb(n+m-1, k, exact=True) - m * comb(n+m-1, k-1, exact=True)) * rate1**k * rate2**(n+m-k)
    
    if m-1 >= 1:
        for k in range(1, m):
            second += (m * comb(n+m-1, k, exact=True) - n * comb(n+m-1, k-1, exact=True)) * rate1**(n+m-k) * rate2**k
    
    num += first + second
    
    return num / denom