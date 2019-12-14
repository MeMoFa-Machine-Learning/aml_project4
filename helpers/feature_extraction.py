import numpy as np


def calculate_percentiles(values):
    n5 = np.percentile(values, 5)
    n25 = np.percentile(values, 25)
    n75 = np.percentile(values, 75)
    n95 = np.percentile(values, 95)
    median = np.percentile(values, 50)
    return n5, n25, median, n75, n95


def calculate_mean_based_stats_full(values):
    mean, std = calculate_mean_based_stats(values)
    var = np.var(values)
    rms = np.mean(np.sqrt(values ** 2))
    return [mean, std, var, rms]


def calculate_mean_based_stats(values):
    mean = np.mean(values)
    std = np.std(values)
    return [mean, std]


def calculate_full_statistics(list_values):
    n5, n25, median, n75, n95 = calculate_percentiles(list_values)
    mean, std, var, rms = calculate_mean_based_stats_full(list_values)
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def max_min_difference(signal):
    return np.max(signal) - np.min(signal)
