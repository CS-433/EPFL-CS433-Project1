# -*- coding: utf-8 -*-
"""
    Script file that produces the same .csv predictions which we used in our best 
    submission to the competition system.
"""

import numpy as np
import zipfile
from pathlib import Path
from proj1_helpers import *
from implementations import *
import warnings


def main():
     
    print('Running...')
    
    # Unzip data files
    my_file = Path("../data/train.csv")
    if not my_file.is_file():
        with zipfile.ZipFile('../data/train.csv.zip', 'r') as zip_ref:
            zip_ref.extractall('../data')

    my_file = Path("../data/test.csv")
    if not my_file.is_file():
        with zipfile.ZipFile('../data/test.csv.zip', 'r') as zip_ref:
            zip_ref.extractall('../data')

    DATA_TRAIN_PATH = '../data/train.csv'
    DATA_TEST_PATH = '../data/test.csv'

    # Load data
    y, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    # Remove related features
    tX_train = remove_related_features(tX_train)
    tX_test = remove_related_features(tX_test)
    
    # Split data
    masks_train = get_masks(tX_train)
    masks_test = get_masks(tX_test)
    
    # Parameters per subset
    lambdas = [0.002, 0.001, 0.001]
    
    # To store predictions
    y_pred = np.zeros(tX_test.shape[0])

    for idx in range(len(masks_train)):
        # Get events
        x_train = tX_train[masks_train[idx]]
        y_train = y[masks_train[idx]]
        x_test = tX_test[masks_test[idx]]

        # Replace missing values
        # We expect to see this RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_train = replace_na_values(x_train)
            x_test = replace_na_values(x_test)
        
        # Obtain weight
        weight, _ = ridge_regression(y_train, x_train, lambdas[idx])

        # Generate predictions
        y_test_pred = predict_labels(weight, x_test)
        y_pred[masks_test[idx]] = y_test_pred
    
    # Generate file csv for submission
    OUTPUT_PATH = '../data/submission.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('Submission file created!')


if __name__ == "__main__":
    main()
