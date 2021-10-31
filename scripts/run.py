import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from proj1_helpers import *


def main():
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    weights, loss_LS = least_squares(y, tX)
    print('f')

    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    OUTPUT_PATH = '../data/submission.csv'
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('f')


if __name__ == "__main__":
    main()