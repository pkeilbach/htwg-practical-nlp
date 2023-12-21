"""Word Embeddings module for HTWG NLP course.

This module contains the WordEmbeddings class for NLP tasks.

Hints:
- Pickle is a Python module for object serialization. You can find more information about it [here](https://docs.python.org/3/library/pickle.html)
- Python context managers are a convenient way to handle resources, like files. You can find more information about them [here](https://book.pythontips.com/en/latest/context_managers.html)
- Note that the norm of a vector can be computed with [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html). Carefully check which arguments you can to pass to the function.
- To find the indices of the `n` smallest or largest values in a numpy array, you can use [numpy.argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html).
"""

import pickle

import numpy as np
import pandas as pd


class WordEmbeddings:
    """Word Embeddings class for NLP tasks.

    This class implements a Word Embeddings model for NLP tasks.

    Attributes:
        embeddings (dict[str, np.ndarray]): the word embeddings. `None` if the embeddings have not been loaded yet.
        embeddings_df (pd.DataFrame): the word embeddings as a DataFrame, where the index is the vocabulary and the columns are the embedding dimensions. `None` if the embeddings have not been loaded yet.
    """

    def __init__(self) -> None:
        """Initializes the WordEmbeddings class."""
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    @property
    def embedding_values(self) -> np.ndarray:
        """Returns the embedding values.

        Returns:
            np.ndarray: the embedding values as a numpy array of shape (n, d), where n is the vocabulary size and d is the number of dimensions
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _load_raw_embeddings(self, path: str) -> None:
        """Loads the raw embeddings from a pickle file, and stores them in the `_embeddings` attribute.

        The embeddings in the pickle file are in the form of a dictionary, where the keys are the words and the values are the embedding vectors as numpy arrays.

        Args:
            path (str): the path to the pickle file
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def _load_embeddings_to_dataframe(self) -> None:
        """Loads the embeddings from the `_embeddings` attribute to the `_embeddings_df` attribute.

        The `_embeddings_df` attribute is a pandas DataFrame, where the index is the vocabulary and the columns are the embedding dimensions.
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def load_embeddings(self, path: str) -> None:
        """Loads the embeddings from a pickle file.

        Args:
            path (str): the path to the pickle file
        """
        self._load_raw_embeddings(path)
        self._load_embeddings_to_dataframe()

    def get_embeddings(self, word: str) -> np.ndarray | None:
        """Returns the embedding vector for a given word.

        Does not raise an exception if the word is not in the vocabulary, but returns None instead.

        Args:
            word (str): the word to get the embedding vector for

        Returns:
            np.ndarray | None: the embedding vector for the given word in the form of a numpy array of shape (d,), where d is the number of dimensions, or None if the word is not in the vocabulary
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def euclidean_distance(self, v: np.ndarray) -> np.ndarray:
        """Returns the Euclidean distance between the given vector `v` and all the embedding vectors.

        Args:
            v (np.ndarray): the vector to compute the distance to

        Returns:
            np.ndarray: the Euclidean distances between the given vector `v` and all the embedding vectors as a one-dimensional numpy array of shape (n,), where n is the vocabulary size
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def cosine_similarity(self, v: np.ndarray) -> np.ndarray:
        """Returns the cosine similarity between the given vector `v` and all the embedding vectors.

        Args:
            v (np.ndarray): the vector to compute the similarity to

        Returns:
            np.ndarray: the cosine similarities between the given vector `v` and all the embedding vectors as a one-dimensional numpy array of shape (n,), where n is the vocabulary size
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def get_most_similar_words(
        self, word: str, n: int = 5, metric: str = "euclidean"
    ) -> list[str]:
        """Returns the `n` most similar words to the given word.

        The similarity is computed using the Euclidean distance or the cosine similarity.

        Note that the word itself must not be included in the returned list.

        Args:
            word (str): the word to get the most similar words for
            n (int, optional): the number of most similar words to return. Defaults to 5.
            metric (str, optional): the metric to use for computing the similarity. Defaults to "euclidean".

        Raises:
            ValueError: if the metric is not "euclidean" or "cosine"
            AssertionError: if the word is not in the vocabulary

        Returns:
            list[str]: the `n` most similar words to the given word
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")

    def find_closest_word(self, v: np.ndarray, metric: str = "euclidean") -> str:
        """Returns the word that is closest to the given vector `v`.

        The similarity is computed using the Euclidean distance or the cosine similarity.

        Args:
            v (np.ndarray): the vector to find the closest word for
            metric (str, optional): the metric to use for computing the similarity. Defaults to "euclidean".

        Raises:
            ValueError: if the metric is not "euclidean" or "cosine"

        Returns:
            str: the word that is closest to the given vector `v`
        """
        # TODO ASSIGNMENT-4: implement this method
        raise NotImplementedError("This method needs to be implemented.")
