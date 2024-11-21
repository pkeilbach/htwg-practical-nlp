from collections import defaultdict

import numpy as np
import pytest

from htwgnlp.features import CountVectorizer

tweets = [
    ["this", "is", "a", "tweet"],  # 1
    ["a", "happy", "tweet"],  # 1
    ["what", "a", "sad", "tweet"],  # 0
]
labels = np.array([[1], [1], [0], [1], [0]])


@pytest.fixture
def vectorizer():
    return CountVectorizer()


@pytest.fixture
def vectorizer_initialized(vectorizer):
    vectorizer.build_word_frequencies(tweets, labels)
    return vectorizer


def test_init(vectorizer):
    assert isinstance(vectorizer.word_frequencies, defaultdict)
    assert vectorizer.word_frequencies["some_non_existent_key"] == 0


def test_build_word_frequencies(vectorizer_initialized):
    assert vectorizer_initialized.word_frequencies[("this", 1)] == 1
    assert vectorizer_initialized.word_frequencies[("is", 1)] == 1
    assert vectorizer_initialized.word_frequencies[("a", 1)] == 2
    assert vectorizer_initialized.word_frequencies[("tweet", 1)] == 2
    assert vectorizer_initialized.word_frequencies[("what", 0)] == 1
    assert vectorizer_initialized.word_frequencies[("a", 0)] == 1
    assert vectorizer_initialized.word_frequencies[("sad", 0)] == 1
    assert vectorizer_initialized.word_frequencies[("tweet", 0)] == 1


@pytest.mark.parametrize(
    "tweet, expected",
    [
        (["this", "is", "a", "test", "tweet"], np.array([1, 6, 2])),
        (["some", "random", "tweet"], np.array([1, 2, 1])),
        (["this", "is", "a", "sunny", "day"], np.array([1, 4, 1])),
        (["this", "is", "a", "sunny", "and", "a", "warm", "day"], np.array([1, 6, 2])),
    ],
)
def test_get_features(vectorizer_initialized, tweet, expected):
    assert np.array_equal(vectorizer_initialized.get_features(tweet), expected)
