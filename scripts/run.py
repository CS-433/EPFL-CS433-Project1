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
    
    # To store predictions
    preds = np.zeros(tX_test.shape[0])
    
    # Parameters
    degree = 4
    lambda_ = 1e-5
    
    for i in range(len(masks_train)):
        # Get events
        train_data = tX_train[masks_train[i]]
        train_y = y[masks_train[i]]
        test_data = tX_test[masks_test[i]]

        # Replace missing values
        # We expect to see this RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            train_data = replace_na_values(train_data)
            test_data = replace_na_values(test_data)
        
        # Build poly
        train_phi = build_poly(train_data, degree)
        test_phi = build_poly(test_data, degree)
        
        # Obtain weight
        weight, _ = ridge_regression(train_y, train_phi, lambda_)

        # Generate predictions
        pred_y = predict_labels(weight, test_phi)
        preds[masks_test[i]] = pred_y
    
    # Generate file csv for submission
    OUTPUT_PATH = '../data/submission.csv'
    create_csv_submission(ids_test, preds, OUTPUT_PATH)
    print('Submission file created!')


if __name__ == "__main__":
    main()
