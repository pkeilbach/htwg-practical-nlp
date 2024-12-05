"""Tests for the embeddings module.
"""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from htwgnlp.embeddings import WordEmbeddings


@pytest.fixture
def embeddings():
    return WordEmbeddings()


@pytest.fixture
def loaded_embeddings(embeddings):
    embeddings._load_raw_embeddings("notebooks/data/embeddings.pkl")
    embeddings._load_embeddings_to_dataframe()
    return embeddings


@pytest.fixture
def test_vector():
    return np.ones(300)


def test_init(embeddings):
    assert embeddings._embeddings is None
    assert embeddings._embeddings_df is None


def test_load_raw_embeddings(loaded_embeddings):
    assert loaded_embeddings._embeddings is not None
    assert isinstance(loaded_embeddings._embeddings, dict)
    assert "happy" in loaded_embeddings._embeddings.keys()
    assert isinstance(loaded_embeddings._embeddings["happy"], np.ndarray)


def test_load_embeddings_to_dataframe(loaded_embeddings):
    assert loaded_embeddings._embeddings_df is not None
    assert isinstance(loaded_embeddings._embeddings_df, pd.DataFrame)
    assert "happy" in loaded_embeddings._embeddings_df.index
    assert isinstance(loaded_embeddings._embeddings_df.loc["happy"], pd.Series)
    assert loaded_embeddings._embeddings_df.shape == (243, 300)


def test_embedding_values(loaded_embeddings):
    assert isinstance(loaded_embeddings.embedding_values, np.ndarray)
    assert loaded_embeddings.embedding_values.shape == (243, 300)


def test_get_embeddings(loaded_embeddings):
    assert isinstance(loaded_embeddings.get_embeddings("happy"), np.ndarray)
    assert loaded_embeddings.get_embeddings("happy").shape == (300,)
    assert loaded_embeddings.get_embeddings("non_existent_word") is None


def test_euclidean_distance(loaded_embeddings, test_vector):
    assert isinstance(loaded_embeddings.euclidean_distance(test_vector), np.ndarray)
    assert loaded_embeddings.euclidean_distance(test_vector).shape == (243,)
    np.testing.assert_allclose(
        loaded_embeddings.euclidean_distance(test_vector)[0], 17.507894003796004
    )
    np.testing.assert_allclose(
        loaded_embeddings.euclidean_distance(test_vector)[1], 17.76195946823725
    )
    np.testing.assert_allclose(
        loaded_embeddings.euclidean_distance(test_vector)[42], 17.787844721963356
    )
    np.testing.assert_allclose(
        loaded_embeddings.euclidean_distance(test_vector)[242], 17.745477284490963
    )


def test_cosine_similarity(loaded_embeddings, test_vector):
    assert isinstance(loaded_embeddings.cosine_similarity(test_vector), np.ndarray)
    assert loaded_embeddings.cosine_similarity(test_vector).shape == (243,)
    np.testing.assert_allclose(
        loaded_embeddings.cosine_similarity(test_vector)[0],
        -0.037310105006509546,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded_embeddings.cosine_similarity(test_vector)[1],
        -0.12679458247346523,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded_embeddings.cosine_similarity(test_vector)[42],
        -0.026496807469057613,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded_embeddings.cosine_similarity(test_vector)[242],
        -0.0657470030012723,
        rtol=1e-5,
        atol=1e-5,
    )


def test_find_closest_word(loaded_embeddings, test_vector):
    assert isinstance(loaded_embeddings.find_closest_word(test_vector), str)
    assert loaded_embeddings.find_closest_word(test_vector) == "Bahamas"


def test_get_most_similar_words(loaded_embeddings):
    assert isinstance(loaded_embeddings.get_most_similar_words("Germany"), list)
    assert loaded_embeddings.get_most_similar_words("Germany") == [
        "Austria",
        "Belgium",
        "Switzerland",
        "France",
        "Hungary",
    ]
    assert loaded_embeddings.get_most_similar_words("Germany", metric="cosine") == [
        "Austria",
        "Switzerland",
        "Hungary",
        "Poland",
        "Berlin",
    ]


def test_get_most_similar_words_with_non_existent_word(loaded_embeddings):
    with pytest.raises(AssertionError):
        loaded_embeddings.get_most_similar_words("non_existent_word")


@pytest.mark.parametrize(
    "metric, expectation",
    [
        ("euclidean", does_not_raise()),
        ("cosine", does_not_raise()),
        ("invalid_metric", pytest.raises(ValueError)),
    ],
)
def test_get_most_similar_words_with_invalid_metric(
    metric, expectation, loaded_embeddings
):
    with expectation:
        loaded_embeddings.get_most_similar_words("Germany", metric=metric)
