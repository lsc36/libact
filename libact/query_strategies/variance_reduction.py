"""Variance Reduction"""
import copy
from multiprocessing import Pool

import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
import libact.models
from libact.query_strategies._variance_reduction import estVar
from libact.utils import inherit_docstring_from, zip


class VarianceReduction(QueryStrategy):

    """Variance Reduction

    This class implements Variance Reduction active learning algorithm [1]_.

    Parameters
    ----------
    model : {libact.model.LogisticRegression instance, 'LogisticRegression'}
        The model used for variance reduction to evaluate the variance.
        Only Logistic regression are supported now.

    sigma : float, >0, optional (default=100.0)
        1/sigma is added to the diagonal of the Fisher information matrix as a
        regularization term.

    optimality : {'trace', 'determinant', 'eigenvalue'},\
            optional (default='trace')
        The type of optimal design.  The options are the trace, determinant, or
        maximum eigenvalue of the inverse Fisher information matrix.
        Only 'trace' are supported now.

    n_jobs : int, optional (default=1)
        The number of processors to estimate the expected variance.

    Attributes
    ----------


    References
    ----------
    .. [1] Schein, Andrew I., and Lyle H. Ungar. "Active learning for logistic
           regression: an evaluation." Machine Learning 68.3 (2007): 235-265.

    .. [2] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = kwargs.pop('model', None)
        if isinstance(model, str):
            self.model = getattr(libact.models, model)()
        else:
            self.model = model
        self.optimality = kwargs.pop('optimality', 'trace')
        self.sigma = kwargs.pop('sigma', 1.0)
        self.n_jobs = kwargs.pop('n_jobs', 1)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        Xlabeled, y = self.dataset.get_labeled_entries()
        Xlabeled = np.array(Xlabeled)
        y = list(y)

        unlabeled_entry_ids, X_pool = self.dataset.get_unlabeled_entries()

        label_count = self.dataset.get_num_of_labels()

        clf = copy.copy(self.model)
        clf.train(Dataset(Xlabeled, y))

        p = Pool(self.n_jobs)
        errors = p.map(_E, [(Xlabeled, y, x, clf, label_count, self.sigma,
                             self.model) for x in X_pool])
        p.terminate()

        return unlabeled_entry_ids[errors.index(min(errors))]


def _Phi(sigma, PI, X, epi, ex, label_count, feature_count):
    ret = estVar(sigma, PI, X, epi, ex)
    return ret


def _E(args):
    X, y, qx, clf, label_count, sigma, model = args
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    query_point = sigmoid(clf.predict_real([qx]))
    feature_count = len(X[0])
    ret = 0.0
    for i in range(label_count):
        clf_ = copy.copy(model)
        clf_.train(Dataset(np.vstack((X, [qx])), np.append(y, i)))
        PI = sigmoid(clf_.predict_real(np.vstack((X, [qx]))))
        ret += query_point[-1][i] * _Phi(sigma, PI[:-1], X, PI[-1], qx,
                                         label_count, feature_count)
    return ret
