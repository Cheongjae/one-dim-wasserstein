import gzip
import os
import os.path
import zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import spikebench.transforms as transforms
from spikebench.dataset_adapters import retina_dataset
from spikebench.encoders import TrainBinarizationTransform

def download_retina(data_path):
    data_path = Path(data_path)

    URL = 'https://drive.google.com/uc?id=1HqZSs7r14bC97gWvw_VJ63CsVl3Ug6DM'
    output = str(data_path / 'retinal_data.zip')
    gdown.download(URL, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    with zipfile.ZipFile(str(data_path / 'error_robust_mode_data.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(output)
    os.remove(str(data_path / 'error_robust_mode_data.zip'))


def load_retina_all(
    dataset_path='./data/retina',
    random_seed=0,
    test_size=0.3,
    n_samples=None,
    window_size=200,
    step_size=200,
    encoding='isi',
    bin_size=80,
):
    states = ['randomly_moving_bar',
                     'repeated_natural_movie',
                     'unique_natural_movie',
                     'white_noise_checkerboard',]

    dataset_path = Path(dataset_path)
    DELIMITER = None

    if not os.path.exists(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)
        download_retina(dataset_path)

    retinal_spike_data = retina_dataset(str(dataset_path / 'mode_paper_data'))
    # return retinal_spike_data
    group_split = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    X = np.hstack(
        [
            retinal_spike_data[states[0]].series.values,
            retinal_spike_data[states[1]].series.values,
            retinal_spike_data[states[2]].series.values,
            retinal_spike_data[states[3]].series.values,
        ]
    )
    y = np.hstack(
        [
            np.zeros(retinal_spike_data[states[0]].shape[0]),
            np.ones(retinal_spike_data[states[1]].shape[0]),
            2*np.ones(retinal_spike_data[states[2]].shape[0]),
            3*np.ones(retinal_spike_data[states[3]].shape[0]),
        ]
    )
    groups = np.hstack(
        [
            retinal_spike_data[states[0]].groups.values,
            retinal_spike_data[states[1]].groups.values,
            retinal_spike_data[states[2]].groups.values,
            retinal_spike_data[states[3]].groups.values,
        ]
    )

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    return encode_spike_trains(
        X_train,
        X_test,
        y_train,
        y_test,
        delimiter=DELIMITER,
        encoding=encoding,
        window_size=window_size,
        step_size=step_size,
        n_samples=n_samples,
        bin_size=bin_size,
    )

def encode_spike_trains(
    X_train,
    X_test,
    y_train,
    y_test,
    delimiter,
    encoding='isi',
    window_size=200,
    step_size=100,
    n_samples=None,
    bin_size=80,
):
    normalizer = transforms.TrainNormalizeTransform(
        window=window_size, step=step_size, n_samples=n_samples
    )
    X_train, y_train, groups_train = normalizer.transform(
        X_train, y_train, delimiter=delimiter
    )
    X_test, y_test, groups_test = normalizer.transform(
        X_test, y_test, delimiter=delimiter
    )
    if encoding == 'sce':
        binarizer = TrainBinarizationTransform(bin_size=bin_size)
        X_train = binarizer.transform(
            pd.DataFrame(
                {
                    'series': [
                        ' '.join([str(v) for v in X_train[idx, :]])
                        for idx in range(X_train.shape[0])
                    ]
                }
            )
        )
        X_test = binarizer.transform(
            pd.DataFrame(
                {
                    'series': [
                        ' '.join([str(v) for v in X_test[idx, :]])
                        for idx in range(X_test.shape[0])
                    ]
                }
            )
        )

    return X_train, X_test, y_train, y_test, groups_train, groups_test