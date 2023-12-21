"""Naive Bayes classifier for NLP.

This module contains the NaiveBayes class for NLP tasks.

Implementing this module is the 3rd assignment of the course. You can find your tasks by searching for `TODO ASSIGNMENT-3` comments.

Hints:
- Find more information about the Python property decorator [here](https://www.programiz.com/python-programming/property)
- To build the word frequencies, you can use the [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) class from Python's collections module
- you may also find the Python [zip](https://docs.python.org/3/library/functions.html#zip) function useful.
- for prediction, you may find the [intersection](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.intersection.html) method of the pandas Index class useful.

"""


from collections import Counter

import numpy as np
import pandas as pd


class NaiveBayes:
    """Naive Bayes classifier for NLP tasks.

    This class implements a Naive Bayes classifier for NLP tasks.
    It can be used for binary classification tasks.

    Attributes:
        word_probabilities (pd.DataFrame): the word probabilities per class, None before training
        df_freqs (pd.DataFrame): the word frequencies per class, None before training
        log_ratios (pd.Series): the log ratios of the word probabilities, None before training
        logprior (float): the logprior of the model, 0 before training
        alpha (float): the smoothing parameter of the model
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initializes the NaiveBayes class.

        The init method accepts one hyperparameter as an optional argument, the smoothing parameter alpha.

        Args:
            alpha (float, optional): the smoothing parameter. Defaults to 1.0.
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    @property
    def logprior(self) -> float:
        """Returns the logprior.

        Returns:
            float: the logprior
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    @logprior.setter
    def logprior(self, y: np.ndarray) -> None:
        """Sets the logprior.

        Note that `y` must contain both classes.

        Args:
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _get_word_frequencies(self, X: list[list[str]], y: np.ndarray) -> None:
        """Computes the word frequencies per class.

        For a given list of tokenized text and a numpy array of class labels, the method computes the word frequencies for each class and stores them as a pandas DataFrame in the `df_freqs` attribute.

        In pandas, if a word does not occur in a class, the frequency should be set to 0, and not to NaN. Also make sure that the frequencies are of type int.

        Note that the this implementation of Naive Bayes is designed for binary classification.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples.
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples.
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _get_word_probabilities(self) -> None:
        """Computes the conditional probabilities of a word given a class using Laplacian Smoothing.

        Based on the word frequencies, the method computes the conditional probabilities for a word given its class and stores them in the `word_probabilities` attribute.
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _get_log_ratios(self) -> None:
        """Computes the log ratio of the conditional probabilities.

        Based on the word probabilities, the method computes the log ratios and stores them in the `log_ratios` attribute.
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def fit(self, X: list[list[str]], y: np.ndarray) -> None:
        """Fits a Naive Bayes model for the given text samples and labels.

        Before training naive bayes, a couple of assertions are performed to check the validity of the input data:
            - The number of text samples and labels must be equal.
            - y must be a 2-dimensional array.
            - y must be a column vector.

        if all assertions pass, the method calls the Naive Bayes training method is executed.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _train_naive_bayes(self, X: list[list[str]], y: np.ndarray) -> None:
        """Trains a Naive Bayes model for the given text samples and labels.

        Training is done in four steps:
            - Compute the log prior ratio
            - Compute the word frequencies
            - Compute the word probabilities of a word given a class using Laplacian Smoothing
            - Compute the log ratios

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples
            y (np.ndarray): a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def predict(self, X: list[list[str]]) -> np.ndarray:
        """Predicts the class labels for the given text samples.

        The class labels are returned as a column vector, where each entry represents the class label of the corresponding sample.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class labels of shape (m, 1), where m is the number of samples
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def predict_prob(self, X: list[list[str]]) -> np.ndarray:
        """Calculates the log likelihoods for the given text samples.

        The class probabilities are returned as a column vector, where each entry represents the probability of the corresponding sample.

        Args:
            X (list[list[str]]): a list of tokenized text samples of length m, where m is the number of samples

        Returns:
            np.ndarray: a numpy array of class probabilities of shape (m, 1), where m is the number of samples
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def predict_single(self, x: list[str]) -> float:
        """Calculates the log likelihood for a single text sample.

        Words that are not in the vocabulary are ignored.

        Args:
            x (list[str]): a tokenized text sample

        Returns:
            float: the log likelihood of the text sample
        """
        # TODO ASSIGNMENT-3: implement this method
        raise NotImplementedError("This method needs to be implemented.")
