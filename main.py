import pandas as pd
import os.path as ospath
import argparse
import numpy as np
from os import makedirs
from itertools import tee
from csv import reader
import biosppy.signals.eeg as eeg
import biosppy.signals.emg as emg
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, butter, lfilter
from tqdm import tqdm
from statistics import median as pymedian
from scipy.stats import entropy as sci_entropy
from collections import Counter, deque
from helpers.helpers import EegStore, EmgStore
from helpers.feature_extraction import *
from imblearn.under_sampling import RandomUnderSampler

import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s')

# Debug parameters
first_n_lines_input = 500


def perform_data_scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def read_in_irregular_csv(path_to_file, skip_n_lines=1, debug=False):
    file_array = []
    with open(path_to_file, 'r') as csv_file:
        brain_waves_reader = reader(csv_file, delimiter=',', quotechar='|')
        for row_to_skip in range(skip_n_lines):
            next(brain_waves_reader)
        for i, row in enumerate(tqdm(brain_waves_reader)):
            if debug and i == first_n_lines_input:
                break
            file_array.append(np.array(row[1:], dtype=np.float32))
    return file_array


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='highpass', analog=False, output='ba')
    return b, a


def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
    normal_lowcut = lowcut / nyq
    normal_highcut = highcut / nyq
    b, a = butter(N=order, Wn=[normal_lowcut, normal_highcut], btype='bandstop', analog=False, output='ba')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_manual_features(eeg1, eeg2, emg1, show_graphs=False):
    manual_features_array = deque()
    for i in tqdm(range(0, len(eeg1))):
        if show_graphs:
            eeg_comb = np.concatenate((eeg1[i].transpose().reshape((eeg1[i].shape[0], 1)),
                                       eeg2[i].transpose().reshape((eeg2[i].shape[0], 1))), axis=1)
            eeg_params = EegStore(*eeg.eeg(signal=eeg_comb, sampling_rate=128, show=show_graphs))

        eeg1_params = EegStore(*eeg.eeg(signal=eeg1[i].transpose().reshape((eeg1[i].shape[0], 1)), sampling_rate=128, show=show_graphs))
        eeg2_params = EegStore(*eeg.eeg(signal=eeg2[i].transpose().reshape((eeg2[i].shape[0], 1)), sampling_rate=128, show=show_graphs))
        # emg_params = EmgStore(*emg.emg(signal=emg1[i], sampling_rate=128, show=False)) TODO: Try to find work-around

        # Adding features
        feature_extracted_samples = (
            *calculate_mean_based_stats(eeg1_params.filtered),
            *calculate_mean_based_stats(eeg2_params.filtered),
            max_min_difference(eeg1_params.filtered),
            max_min_difference(eeg2_params.filtered),
            max_min_difference(emg1)
        )

        manual_features_array.append(feature_extracted_samples)
    return np.array(manual_features_array)


def down_sample_all_channels(eeg1, eeg2, emg, y_train):
    rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    rus.fit_resample(eeg1, y_train)
    sample_indices_sorted = np.sort(rus.sample_indices_)
    eeg1 = eeg1[sample_indices_sorted]
    eeg2 = eeg2[sample_indices_sorted]
    emg = emg[sample_indices_sorted]
    y_train = y_train[sample_indices_sorted]
    return eeg1, eeg2, emg, y_train, np.argmax(sample_indices_sorted >= 43200)


def train_test_split_by_individual(x, y, indivdual_3_cutoff_i, debug=False):
    if debug:
        hold_out_start_i = int(x.shape[0] * 2 / 3)
    else:
        hold_out_start_i = indivdual_3_cutoff_i
    x_gs, y_gs = x[:hold_out_start_i, :], y[:hold_out_start_i]
    x_ho, y_ho = x[hold_out_start_i:, :], y[hold_out_start_i:]
    return x_gs, y_gs, x_ho, y_ho


def main(debug=False, show_graphs=False, outfile="out.csv"):
    output_pathname = "output"
    output_filepath = ospath.join(output_pathname, outfile)
    training_data_dir = ospath.join("data", "training")
    testing_data_dir = ospath.join("data", "testing")

    # Load training data
    logging.info("Reading in training data...")
    train_data_eeg1 = read_in_irregular_csv(ospath.join(training_data_dir, "train_eeg1.csv"), debug=debug)
    train_data_eeg2 = read_in_irregular_csv(ospath.join(training_data_dir, "train_eeg2.csv"), debug=debug)
    train_data_emg = read_in_irregular_csv(ospath.join(training_data_dir, "train_emg.csv"), debug=debug)
    train_data_y = pd.read_csv(ospath.join(training_data_dir, "train_labels.csv"), delimiter=",")["y"]
    if debug:
        train_data_y = train_data_y.head(first_n_lines_input)
    y_train_orig = train_data_y.values
    logging.info("Finished reading in data.")

    # Pre-processing step: mean subtraction
    eeg1_mean = np.mean(train_data_eeg1)
    train_data_eeg1 -= eeg1_mean

    eeg2_mean = np.mean(train_data_eeg2)
    train_data_eeg2 -= eeg2_mean

    emg_mean = np.mean(train_data_emg)
    train_data_emg -= emg_mean

    # Perform undersampling
    (train_data_eeg1,
     train_data_eeg2,
     train_data_emg,
     y_train_ds,
     indivdual_3_cutoff_i) = down_sample_all_channels(train_data_eeg1, train_data_eeg2, train_data_emg, y_train_orig)

    # Pre-processing step: Butterworth filtering
    logging.info("Butterworth filtering...")
    train_data_eeg1 = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), train_data_eeg1))
    train_data_eeg1 = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), train_data_eeg1))
    train_smoothed_eeg1 = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), train_data_eeg1))
    # train_smoothed_eeg1 = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), train_data_eeg1)) TODO: Fix bug

    train_data_eeg2 = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), train_data_eeg2))
    train_data_eeg2 = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), train_data_eeg2))
    train_smoothed_eeg2 = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), train_data_eeg2))
    # train_smoothed_eeg2 = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), train_data_eeg2)) TODO: Fix bug

    train_data_emg = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), train_data_emg))
    train_data_emg = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), train_data_emg))
    train_smoothed_emg = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), train_data_emg))
    # train_smoothed_emg = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), train_data_emg)) TODO: Fix bug
    logging.info("Finished Butterworth filtering")

    # Extract features of training set
    logging.info("Extracting features...")
    x_train_fsel = extract_manual_features(train_smoothed_eeg1, train_smoothed_eeg2, train_smoothed_emg, show_graphs=show_graphs)
    logging.info("Finished extracting features")

    # Load raw ECG testing data
    logging.info("Reading in testing data...")
    test_data_eeg1 = read_in_irregular_csv(ospath.join(testing_data_dir, "test_eeg1.csv"), debug=debug)
    test_data_eeg2 = read_in_irregular_csv(ospath.join(testing_data_dir, "test_eeg2.csv"), debug=debug)
    test_data_emg = read_in_irregular_csv(ospath.join(testing_data_dir, "test_emg.csv"), debug=debug)
    logging.info("Finished reading in data.")

    # Pre-processing step: mean subtraction
    eeg1_mean = np.mean(test_data_eeg1)
    test_data_eeg1 -= eeg1_mean

    eeg2_mean = np.mean(test_data_eeg2)
    test_data_eeg2 -= eeg2_mean

    emg_mean = np.mean(test_data_emg)
    test_data_emg -= emg_mean

    # Pre-processing step: Butterworth filtering
    test_data_eeg1 = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), test_data_eeg1))
    test_data_eeg1 = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), test_data_eeg1))
    test_smoothed_eeg1 = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), test_data_eeg1))
    # test_smoothed_eeg1 = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), test_data_eeg1)) TODO: Fix bug

    test_data_eeg2 = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), test_data_eeg2))
    test_data_eeg2 = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), test_data_eeg2))
    test_smoothed_eeg2 = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), test_data_eeg2))
    # test_smoothed_eeg2 = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), test_data_eeg2)) TODO: Fix bug

    test_data_emg = list(map(lambda x: butter_lowpass_filter(x, 50, 128, 3), test_data_emg))
    test_data_emg = list(map(lambda x: butter_highpass_filter(x, .5, 128, 3), test_data_emg))
    test_smoothed_emg = list(map(lambda x: butter_bandstop_filter(x, 47, 53, 128, 3), test_data_emg))
    # test_smoothed_emg = list(map(lambda x: butter_bandstop_filter(x, 97, 103, 128, 3), test_data_emg)) TODO: Fix bug

    # Extract features of testing set
    logging.info("Extracting features...")
    x_test_fsel = extract_manual_features(test_smoothed_eeg1, test_smoothed_eeg2, test_smoothed_emg, show_graphs=show_graphs)
    logging.info("Finished extracting features")

    # Pre-processing step for meta-feature calculation: StandardScaler
    x_train_fsel, x_test_fsel = perform_data_scaling(x_train_fsel, x_test_fsel)

    # Grid search
    max_depth = [3] if debug else [7, 9, 11, ]
    min_samples_split = [5] if debug else [2, 3, 4, 6, 8]
    n_estimators = [6] if debug else [50, 100, 200, 350, 500]

    knn_neighbors = [3] if debug else [3, 5, 7]
    knn_weights = ['uniform'] if debug else ['uniform', 'distance']
    knn_algorithm = ['brute'] if debug else ['kd_tree', ]
    knn_p = [2] if debug else [1, 2, 3]
    knn_leaf_size = [30] if debug else [20, 30, 40]

    k_best_features = [x_train_fsel.shape[1]] if debug else list(
        np.linspace(start=2, stop=x_train_fsel.shape[1], num=5, endpoint=True, dtype=int))

    models = [
        {
            'model': RandomForestClassifier,
            'parameters': {
                'fs__k': [x_train_fsel.shape[1]],
                'cm__criterion': ['entropy', 'gini'],
                'cm__max_depth': max_depth,
                'cm__min_samples_split': min_samples_split,
                'cm__n_estimators': n_estimators,
                'cm__class_weight': ['balanced'],
            }
        },
        {
            'model': KNC,
            'parameters': {
                'fs__k': k_best_features,
                'cm__n_neighbors': knn_neighbors,
                'cm__weights': knn_weights,
                'cm__algorithm': knn_algorithm,
                'cm__leaf_size': knn_leaf_size,
                'cm__p': knn_p
            }
        }
    ]

    # Perform cross-validation
    x_train_gs, y_train_gs, x_ho, y_ho = train_test_split_by_individual(x_train_fsel, y_train_ds, indivdual_3_cutoff_i, debug=debug)

    best_models = []
    for model in models:
        pl = Pipeline([('fs', SelectKBest()), ('cm', model['model']())], memory=".")
        kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=6)

        # C-support vector classification
        grid_search = GridSearchCV(pl, model['parameters'], scoring="balanced_accuracy", n_jobs=-1, cv=kfold, verbose=1)
        grid_result = grid_search.fit(x_train_gs, y_train_gs)
        # Calculate statistics and calculate on hold-out
        logging.info(
            "Best for model %s: %f using %s" % (str(model['model']), grid_result.best_score_, grid_result.best_params_))
        y_ho_pred = grid_search.predict(x_ho)
        hold_out_score = balanced_accuracy_score(y_ho_pred, y_ho)
        best_models.append((hold_out_score, grid_result.best_params_, model['model']))
        logging.info("Best score on hold-out: {}".format(hold_out_score))

    # Pick best params
    final_model_params_i = int(np.argmax(np.array(best_models)[:, 0]))
    final_model_type = best_models[final_model_params_i][2]
    final_model_params = best_models[final_model_params_i][1]
    logging.info("Picked the following model {} with params: {}".format(str(final_model_type), final_model_params))

    # Fit final model
    logging.info("Fitting the final model...")
    final_model = Pipeline([('fs', SelectKBest()), ('cm', final_model_type())])
    final_model.set_params(**final_model_params)
    final_model.fit(x_train_gs, y_train_gs)

    # Do the prediction
    y_predict = final_model.predict(x_test_fsel)
    unique_elements, counts_elements = np.unique(y_predict, return_counts=True)
    print("test set labels and their corresponding counts")
    print(np.asarray((unique_elements, counts_elements)))

    # Prepare results dataframe
    results = np.zeros((x_test_fsel.shape[0], 2))
    results[:, 0] = list(range(x_test_fsel.shape[0]))
    results[:, 1] = y_predict

    # Save the output weights
    if not ospath.exists(output_pathname):
        makedirs(output_pathname)
    np.savetxt(output_filepath, results, fmt=["%.0f", "%.0f"], newline="\n", delimiter=",", header="Id,y",
               comments="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sleep data")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--show_graphs", action='store_true')
    parser.add_argument("--outfile", required=False, default="out.csv")
    args = parser.parse_args()

    main(debug=args.debug, show_graphs=args.show_graphs, outfile=args.outfile)
