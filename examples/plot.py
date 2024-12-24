#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import LogisticRegression
from libact.query_strategies import RandomSampling, UncertaintySampling, VarianceReduction, HintSVM
from libact.labelers import IdealLabeler


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        lb = lbr.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 10      # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    trn_ds4 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

    # qs3 = VarianceReduction(trn_ds3, model=LogisticRegression())
    # E_in_3, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota)

    qs4 = HintSVM(trn_ds4)
    E_in_4, E_out_4 = run(trn_ds4, tst_ds, lbr, model, qs4, quota)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein',
             linestyle='dashed')
    plt.plot(query_num, E_out_1, 'b', label='qs Eout')
    plt.plot(query_num, E_in_2, 'r', label='random Ein',
             linestyle='dashed')
    plt.plot(query_num, E_out_2, 'r', label='random Eout')
    # plt.plot(query_num, E_in_3, 'g', label='vr Ein',  linestyle='dashed')
    # plt.plot(query_num, E_out_3, 'g', label='vr Eout')
    plt.plot(query_num, E_in_4, 'k', label='SVM Ein',
             linestyle='dashed')
    plt.plot(query_num, E_out_4, 'k', label='SVM Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig("test.png")


if __name__ == '__main__':
    main()
