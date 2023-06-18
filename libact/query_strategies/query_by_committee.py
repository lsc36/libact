"""Query by committee

This module contains a class that implements Query by committee active learning
algorithm.
"""
from __future__ import division

import logging
import math

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ProbabilisticModel
import libact.models
from libact.utils import inherit_docstring_from, seed_random_state, zip

LOGGER = logging.getLogger(__name__)


class QueryByCommittee(QueryStrategy):

    r"""Query by committee

    Parameters
    ----------
    models : list of :py:mod:`libact.models` instances or str
        This parameter accepts a list of initialized libact Model instances,
        or class names of libact Model classes to determine the models to be
        included in the committee to vote for each unlabeled instance.

    disagreement : ['vote', 'kl_divergence'], optional (default='vote')
        Sets the method for measuring disagreement between models.
        'vote' represents vote entropy.
        kl_divergence requires models being ProbabilisticModel

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    students : list, shape = (len(models))
        A list of the model instances used in this algorithm.

    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a QueryByCommittee query_strategy object:

    .. code-block:: python

       from libact.query_strategies import QueryByCommittee
       from libact.models import LogisticRegression

       qs = QueryByCommittee(
                dataset, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1),
                ],
            )


    References
    ----------
    .. [1] Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky. "Query by
           committee." Proceedings of the fifth annual workshop on
           Computational learning theory. ACM, 1992.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.disagreement = kwargs.pop('disagreement', 'vote')


        models = kwargs.pop('models', None)
        if models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
            )
        elif not models:
            raise ValueError("models list is empty")

        if self.disagreement == 'kl_divergence':
            for model in models:
                if not isinstance(model, ProbabilisticModel):
                    raise TypeError(
                        "Given disagreement set as 'kl_divergence', all models"
                        "should be ProbabilisticModel."
                    )


        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.students = list()
        for model in models:
            if isinstance(model, str):
                self.students.append(getattr(libact.models, model)())
            else:
                self.students.append(model)
        self.n_students = len(self.students)
        self.teach_students()

    def _vote_disagreement(self, votes):
        """
        Return the disagreement measurement of the given number of votes.
        It uses the vote vote to measure the disagreement.

        Parameters
        ----------
        votes : list of int, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.

        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The vote entropy of the given votes.
        """
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= lab_count[lab] / self.n_students * \
                    math.log(float(lab_count[lab]) / self.n_students)

        return ret

    def _kl_divergence_disagreement(self, proba):
        """
        Calculate the Kullback-Leibler (KL) divergence disaagreement measure.

        Parameters
        ----------
        proba : array-like, shape=(n_samples, n_students, n_class)

        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The kl_divergence of the given probability.
        """
        n_students = np.shape(proba)[1]
        consensus = np.mean(proba, axis=1) # shape=(n_samples, n_class)
        # average probability of each class across all students
        consensus = np.tile(consensus, (n_students, 1, 1)).transpose(1, 0, 2)
        kl = np.sum(proba * np.log(proba / consensus), axis=2)
        return np.mean(kl, axis=1)

    def _labeled_uniform_sample(self, sample_size):
        """sample labeled entries uniformly"""
        X, y = self.dataset.get_labeled_entries()
        samples_idx = [self.random_state_.randint(0, X.shape[0]) for _ in range(sample_size)]
        return Dataset(X[samples_idx], np.array(y)[samples_idx])

    def teach_students(self):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        dataset = self.dataset
        for student in self.students:
            bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
                LOGGER.warning('There is student receiving only one label,'
                               're-sample the bag.')
            student.train(bag)

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        # Train each model with newly updated label.
        self.teach_students()

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeros((len(X_pool), len(self.students)))
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)

            vote_entropy = self._vote_disagreement(votes)
            ask_idx = self.random_state_.choice(
                    np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])

        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)

            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = self.random_state_.choice(
                    np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])

        return unlabeled_entry_ids[ask_idx]
