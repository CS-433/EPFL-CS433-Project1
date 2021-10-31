# -*- coding: utf-8 -*-
"""
    Script file that produces the same .csv predictions which we used in our best 
    submission to the competition system.
"""

# Useful starting lines
import numpy as np
import zipfile
from pathlib import Path
from proj1_helpers import *
from implementations import least_square


def main():
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

    lambda_ = 1e-5
    weight, _ = ridge_regression(y, tX, lambda_)
    y_pred = predict_labels(weight, tX_test)
        
    # We give the name of the output file
    OUTPUT_PATH = '../data/submission.csv'
    # Generate predictions and save ouput in csv format for submission:
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

    print('Done !')


if __name__ == "__main__":
    main()
