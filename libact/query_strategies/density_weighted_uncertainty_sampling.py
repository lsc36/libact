"""Density Weighted Uncertainty Sampling (DWUS)
"""
from __future__ import division

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip


class DWUS(QueryStrategy):
    """Density Weighted Uncertainty Sampling (DWUS)

    We use the KMeans algorithm for clustering instead of the Kmediod for now.

    Support binary case and LogisticRegression only.

    Parameters
    ----------
    n_clusters : int, optional, default: 5
        Number of clusters for kmeans to cluster.

    sigma : float, optional, default: .1
        The variance of the multivariate gaussian used to model density.

    max_iter : int, optional, default: 100
        The maximum number of iteration used in estimating density through EM
        algorithm.

    tol : float, default: 1e-4
        Tolerance with regards to inertia to declare convergence.

    C : float, default: 1.
        Regularization term for logistic regression.

    kmeans_param : dict, default: {}
        Parameter for sklearn.cluster.KMeans.
        see, http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    kmeans_ : sklearn.cluster.KMeans object
        The clustering algorithm instance.

    p_x : ndarray, shape=(n_labeled + n_unlabeled, )
        The density estimate for each x. Its order is the same as dataset.data.

    Examples
    --------
    Here is an example of how to declare a DWUS query_strategy object:

    .. code-block:: python

       from libact.query_strategies import DWUS
       from libact.models import LogisticRegression

       qs = DWUS(dataset)

    References
    ----------
    .. [1] Donmez, Pinar, Jaime G. Carbonell, and Paul N. Bennett. "Dual
           strategy active learning." Machine Learning: ECML 2007. Springer
           Berlin Heidelberg, 2007. 116-127.
    .. [2] Nguyen, Hieu T., and Arnold Smeulders. "Active learning using
           pre-clustering." Proceedings of the twenty-first international
           conference on Machine learning. ACM, 2004.
    """

    def __init__(self, *args, **kwargs):
        super(DWUS, self).__init__(*args, **kwargs)
        self.n_clusts = kwargs.pop('n_clusters', 5)
        self.sigma = kwargs.pop('sigma', 0.1)
        self.max_iter = kwargs.pop('max_iter', 100)
        self.tol = kwargs.pop('tol', 1e-4)
        self.C = kwargs.pop('C', 1.)
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)
        kmeans_param = kwargs.pop('kmeans_param', {})
        if 'random_state' not in kmeans_param:
            kmeans_param['random_state'] = self.random_state_

        self.kmeans_ = KMeans(n_clusters=self.n_clusts,
                              **kmeans_param)
        all_x = np.array([xy[0] for xy in self.dataset.data])

        # Cluster the data.
        self.kmeans_.fit(all_x)
        d = len(all_x[0])

        centers = self.kmeans_.cluster_centers_
        P_k = np.ones(self.n_clusts) / float(self.n_clusts)

        dis = np.zeros((len(all_x), self.n_clusts))
        for i in range(self.n_clusts):
            dis[:, i] = np.exp(-np.einsum('ij,ji->i', (all_x - centers[i]),
                (all_x - centers[i]).T) / 2 / self.sigma)

        # EM percedure to estimate the prior
        for _ in range(self.max_iter):
            # E-step P(k|x)
            temp = dis * np.tile(P_k, (len(all_x), 1))
            # P_k_x, shape = (len(all_x), n_clusts)
            P_k_x = temp / np.tile(np.sum(temp, axis=1), (self.n_clusts, 1)).T

            # M-step
            P_k = 1./len(all_x) * np.sum(P_k_x, axis=0)

        self.P_k_x = P_k_x

        p_x_k = np.zeros((len(all_x), self.n_clusts))
        for i in range(self.n_clusts):
            p_x_k[:, i] = multivariate_normal.pdf(
                all_x, mean=centers[i], cov=np.ones(d)*np.sqrt(self.sigma))

        self.p_x = np.dot(p_x_k, P_k).reshape(-1)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        unlabeled_entry_ids, _ = self.dataset.get_unlabeled_entries()
        labeled_entry_ids = np.array([eid
                                      for eid, x in enumerate(self.dataset.data)
                                      if x[1] != None])
        labels = np.array([x[1]
                           for eid, x in enumerate(self.dataset.data)
                           if x[1] != None]).reshape(-1, 1)
        centers = self.kmeans_.cluster_centers_
        P_k_x = self.P_k_x
        p_x = self.p_x[list(unlabeled_entry_ids)]

        clf = DensityWeightedLogisticRegression(P_k_x[labeled_entry_ids, :],
                                                centers,
                                                self.C)
        clf.train(labeled_entry_ids, labels)
        P_y_k = clf.predict()

        P_y_x = np.zeros(len(unlabeled_entry_ids))
        for k, center in enumerate(centers):
            P_y_x += P_y_k[k] * P_k_x[unlabeled_entry_ids, k]

        # binary case
        expected_error = P_y_x
        expected_error[P_y_x >= 0.5] = 1. - P_y_x[P_y_x >= 0.5]

        ask_id = np.argmax(expected_error * p_x)

        return unlabeled_entry_ids[ask_id]

class DensityWeightedLogisticRegression(object):
    r"""Density Weighted Logistic Regression

    Density Weighted Logistice Regression is used in DWUS to estimate the
    probability of representing which label for each cluster.
    Density Weighted Logistic Regression optimizes the following likelihood
    function.

    .. math::

        \sum_{i\in I_l} \ln P(y_i|\mathbf{x}_i; w)

    Including the regularization term and
    :math:`P(y,k|x) = \sum^K_{k=1}P(y|k)P(k|x)`, it becomes the following
    function:

    .. math::

        \frac{C}{2} \|w\|2 - \sum_{i\in I_l} \ln \{\sum^K_{k=1} P(k|\mathbf{x}_i) P(y_i|k; w)\}

    Where :math:`K` is the number of clusters, :math:`I_l` is the indices for
    labled data, :math:`w` is the logistice regression parameter,
    :math:`\mathbf{x}_i` and :math`y_i` are the feature vector and label for
    indice :math:`i`.

    Parameters
    ----------
    density_estimate: array-like, shape=(n_samples, n_clusters)
        The probability of each sample to each cluster.

    centers : array-like, shape=(n_clusters, n_features)
        The point of each cluster center.

    C : float
        Regularization term for logistic regression.

    Attributes
    ----------
    self.w_ : ndarray, shape=(n_features + 1, )
        Logistic regression parameter, the last element is the bias term.
    """

    def __init__(self, density_estimate, centers, C):
        self.density = np.asarray(density_estimate)
        self.centers = np.asarray(centers)
        self.C = C
        self.w_ = None

    def _likelihood(self, w, X, y):
        w = w.reshape(-1, 1)
        sigmoid = lambda t: 1. / (1. + np.exp(-t))
        # w --> shape = (d+1, 1)
        L = lambda w: (self.C/2. * np.dot(w[:-1].T, w[:-1]) - \
                np.sum(np.log(
                    np.sum(self.density *
                        sigmoid(np.dot(y,
                                       (np.dot(self.centers, w[:-1]) + w[-1]).T)
                        ), axis=1)
                ), axis=0))[0][0]

        return L(w)

    def train(self, X, y):
        d = np.shape(self.centers)[1]
        w = np.zeros((d+1, 1))
        # TODO Use more sophistic optimization methods
        result = minimize(lambda _w: self._likelihood(_w, X, y),
                          w.reshape(-1),
                          method='CG')
        w = result.x.reshape(-1, 1)

        self.w_ = w

    def predict(self):
        """
        Returns
        -------
        proba : ndarray, shape=(n_clusters, )
            The probability of given cluster being label 1.

        """
        if self.w_ is not None:
            sigmoid = lambda t: 1. / (1. + np.exp(-t))
            return sigmoid(np.dot(self.centers, self.w_[:-1]) + self.w_[-1])
        else:
            # TODO the model is not trained
            pass
