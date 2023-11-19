from collections import defaultdict

import numpy as np


class CountVectorizer:
    def __init__(self):
        """Initializes the CountVectorizer.

        The word_frequencies attribute is a dictionary of word frequencies by class.

        Makes use of [Python's defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict) to initialize the dictionary with 0 for each key.
        This means that if a key is not found in the dictionary, the value is 0 and no KeyError exception is raised.

        """
        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def build_word_frequencies(
        self, tweets: list[list[str]], labels: list[str]
    ) -> None:
        """Builds a dictionary of word frequencies by counting the number of occurrences of each word in each class.

        The key is a tuple of the word and the class, the value is the frequency of the word in the class.
        For example, the key ('happy', 1) is the word 'happy' in the positive class, the value is the number of times the word 'happy' occurs in the positive class.

        Args:
            tweets (list[list[str]]): a list of tokenized tweets
            labels (list[str]): a list of corresponding class labels

        """
        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def get_features(self, tweet: list[str]) -> np.ndarray:
        """Returns a feature vector for a given tweet.

        The feature vector is a row vector of the form [1, pos, neg], where pos is the number of positive words in the tweet and neg is the number of negative words in the tweet.

        Args:
            tweet (list[str]): a tokenized tweet

        Returns:
            np.ndarray: the feature vector for the tweet as a row vector
        """

        # TODO ASSIGNMENT-2: implement this method
        raise NotImplementedError("This method needs to be implemented.")
