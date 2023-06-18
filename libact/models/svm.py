"""SVM

An interface for scikit-learn's C-Support Vector Classifier model.
"""
import logging

import numpy as np
import sklearn.svm
from sklearn.multiclass import OneVsRestClassifier

from libact.base.interfaces import ContinuousModel


LOGGER = logging.getLogger(__name__)


class SVM(ContinuousModel):

    """C-Support Vector Machine Classifier

    When decision_function_shape == 'ovr', we use OneVsRestClassifier(SVC) from
    sklearn.multiclass instead of the output from SVC directory since it is not
    exactly the implementation of One Vs Rest.

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self, *args, **kwargs):
        self.model = sklearn.svm.SVC(*args, **kwargs)
        if self.model.decision_function_shape == 'ovr':
            self.decision_function_shape = 'ovr'
            # sklearn's ovr isn't real ovr
            self.model = OneVsRestClassifier(self.model)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T

        if self.decision_function_shape != 'ovr':
            LOGGER.warning("SVM model support only 'ovr' for multiclass"
                           "predict_real.")
        return dvalue
