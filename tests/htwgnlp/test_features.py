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
    vectorizer = CountVectorizer()
    vectorizer.build_word_frequencies(tweets, labels)
    return vectorizer


def test_init():
    vec = CountVectorizer()
    assert isinstance(vec.word_frequencies, defaultdict)
    assert vec.word_frequencies["some_non_existent_key"] == 0


def test_build_word_frequencies(vectorizer):
    assert vectorizer.word_frequencies[("this", 1)] == 1
    assert vectorizer.word_frequencies[("is", 1)] == 1
    assert vectorizer.word_frequencies[("a", 1)] == 2
    assert vectorizer.word_frequencies[("tweet", 1)] == 2
    assert vectorizer.word_frequencies[("what", 0)] == 1
    assert vectorizer.word_frequencies[("a", 0)] == 1
    assert vectorizer.word_frequencies[("sad", 0)] == 1
    assert vectorizer.word_frequencies[("tweet", 0)] == 1


@pytest.mark.parametrize(
    "tweet, expected",
    [
        (["this", "is", "a", "test", "tweet"], np.array([1, 6, 2])),
        (["some", "random", "tweet"], np.array([1, 2, 1])),
        (["this", "is", "a", "sunny", "day"], np.array([1, 4, 1])),
        (["this", "is", "a", "sunny", "and", "a", "warm", "day"], np.array([1, 6, 2])),
    ],
)
def test_get_features(vectorizer, tweet, expected):
    assert np.array_equal(vectorizer.get_features(tweet), expected)
