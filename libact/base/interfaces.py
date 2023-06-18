"""
Base interfaces for use in the package.
The package works according to the interfaces defined below.
"""
from six import with_metaclass

from abc import ABCMeta, abstractmethod


class QueryStrategy(with_metaclass(ABCMeta, object)):

    """Pool-based query strategy

    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._dataset

    def update(self, entry_id, label):
        """Update the internal states of the QueryStrategy after each queried
        sample being labeled.

        Parameters
        ----------
        entry_id : int
            The index of the newly labeled sample.

        label : float
            The label of the queried sample.
        """

    def _get_scores(self):
        """Return the score used for making query, the larger the better. Read-only.

        No modification to the internal states.

        Returns
        -------
        (ask_id, scores): list of tuple (int, float)
            The index of the next unlabeled sample to be queried and the score assigned.
        """

    @abstractmethod
    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
        """


class Labeler(with_metaclass(ABCMeta, object)):

    """Label the queries made by QueryStrategies

    Assign labels to the samples queried by QueryStrategies.
    """
    @abstractmethod
    def label(self, feature):
        """Return the class labels for the input feature array.

        Parameters
        ----------
        feature : array-like, shape (n_features,)
            The feature vector whose label is to queried.

        Returns
        -------
        label : int
            The class label of the queried feature.
        """


class Model(with_metaclass(ABCMeta, object)):

    """Classification Model

    A Model returns a class-predicting function for future samples after
    trained on a training dataset.
    """
    @abstractmethod
    def train(self, dataset, *args, **kwargs):
        """Train a model according to the given training dataset.

        Parameters
        ----------
        dataset : Dataset object
             The training dataset the model is to be trained on.

        Returns
        -------
        self : object
            Returns self.
        """

    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        """Predict the class labels for the input samples

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The unlabeled samples whose labels are to be predicted.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            The class labels for samples in the feature array.
        """

    @abstractmethod
    def score(self, testing_dataset, *args, **kwargs):
        """Return the mean accuracy on the test dataset

        Parameters
        ----------
        testing_dataset : Dataset object
            The testing dataset used to measure the perforance of the trained
            model.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """


class MultilabelModel(Model):
    """Multilabel Classification Model

    A Model returns a multilabel-predicting function for future samples after
    trained on a training dataset.
    """


class ContinuousModel(Model):

    """Classification Model with intermediate continuous output

    A continuous classification model is able to output a real-valued vector
    for each features provided.
    """
    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        """Predict confidence scores for samples.

        Returns the confidence score for each (sample, class) combination.

        The larger the value for entry (sample=x, class=k) is, the more
        confident the model is about the sample x belonging to the class k.

        Take Logistic Regression as example, the return value is the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose confidence scores are to be predicted.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the confidence scores per (sample, class)
            combination.
        """


class ProbabilisticModel(ContinuousModel):

    """Classification Model with probability output

    A probabilistic classification model is able to output a real-valued vector
    for each features provided.
    """
    def predict_real(self, feature, *args, **kwargs):
        return self.predict_proba(feature, *args, **kwargs)

    @abstractmethod
    def predict_proba(self, feature, *args, **kwargs):
        """Predict probability estimate for samples.

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose probability estimation are to be predicted.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the prabablity estimate for each class.
        """
