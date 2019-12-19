import numpy as np
import scipy as sp
from scipy.stats import kurtosis, skew


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


#######################################
####### Frequency Features ############
#######################################

def extract_peaks(signal):
    """returns peaks and properties
    Args:
        signal (1D-array): frequency transformed signal
    Returns:
        1D-array, dict: indices of peaks, {‘peak_heights’, ‘left_thresholds’, ‘right_thresholds’, ‘prominences’, ‘right_bases’, ‘left_bases’, ‘width_heights’, ‘left_ips’, ‘right_ips’, ‘plateau_sizes’, left_edges’, ‘right_edges’}
    """
    peaks = sp.signal.find_peaks(signal, height=0)
    return peaks

def get_dominant_peaks_position(peak_positions, peak_dict):
    """get the position of peaks (ordered by height) from the return values of extract_peaks()
    Args:
        peak_positions (int): indices of the peak positions in the signal
        peak_dict (dict): dict with peak information from extract_peaks()
    Returns:
        1D-array: List with position of peaks ordered by descending height
    """
    peak_array = np.zeros((2, len(peak_positions))) # array with peak positions and respective heights
    peak_array[0] = peak_positions
    peak_array[1] = peak_dict["peak_heights"]
    sorted_desc = peak_array[:, peak_array[1].argsort()]
    return sorted_desc[0]


def calculate_skew_kurtosis_difference(signal1, signal2):
    skew_difference = skew(signal1, bias=False) - skew(signal2, bias=False)
    kurtosis_difference = kurtosis(signal1, bias=False) - kurtosis(signal2, bias=False)
    return [skew_difference[0], kurtosis_difference[0]]


def calculate_skew_kurtosis_difference_emg(signal1, signal2):
    skew_difference = skew(signal1, bias=False) - skew(signal2, bias=False)
    kurtosis_difference = kurtosis(signal1, bias=False) - kurtosis(signal2, bias=False)
    return [skew_difference, kurtosis_difference]
