"""feature module for NLP.

This module contains the CountVectorizer class for NLP tasks.
"""

from collections import defaultdict

import numpy as np
import numpy.typing as npt


class CountVectorizer:
    """CountVectorizer class for NLP tasks.

    This class implements a CountVectorizer for NLP tasks.
    It can be used to build a feature vector for a given tweet.

    Attributes:
        word_frequencies (defaultdict[tuple[str, int], int]): a dictionary of word frequencies by class
    """

    def __init__(self):
        """Initializes the CountVectorizer.

        The `word_frequencies` attribute is a dictionary of word frequencies by class.

        Makes use of [Python's defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict) to initialize the dictionary with 0 for each key.
        This means that if a key is not found in the dictionary, the value is 0 and no KeyError exception is raised.

        """
        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def build_word_frequencies(
        self, tweets: list[list[str]], labels: npt.NDArray[np.int_]
    ) -> None:
        """Builds a dictionary of word frequencies by counting the number of occurrences of each word in each class.

        The key is a tuple of the word and the class, the value is the frequency of the word in the class.
        For example, the key ('happi', 1) refers to the word 'happi' in the positive class,
        while ('happi', 0) refers to 'happi' in the negative class,
        and the value it holds is the number of times the word 'happi' occurs in the positive class.

        Example:
            >>> vec = CountVectorizer()
            >>> vec.build_word_frequencies(tweets, labels)
            >>> vec.word_frequencies[('happi', 1)]
            42

        Args:
            tweets (list[list[str]]): a list of tokenized tweets
            labels (npt.NDArray[np.int_]): a column vector of corresponding class labels , e.g. `np.array([[1], [0], [1], [0]]`)
        """
        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def get_features(self, tweet: list[str]) -> np.ndarray:
        """Returns a feature vector for a given tweet.

        The feature vector is a row vector of the form `[1, pos, neg]`, where `pos` and `neg` are the sum of the frequencies of the words in the tweet in the positive and negative class, respectively.

        Args:
            tweet (list[str]): a tokenized tweet

        Returns:
            np.ndarray: the feature vector for the tweet as a row vector
        """

        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")
