from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from htwgnlp.naive_bayes import NaiveBayes

model = NaiveBayes()

train_data = {
    "samples": [
        ["I", "am", "happy"],
        ["I", "am", "sad"],
        ["I", "am", "happy", "because", "I", "love", "the", "weather"],
        ["I", "am", "sad", "because", "I", "hate", "the", "weather"],
    ],
    "labels": np.array([1, 0, 1, 0]).reshape(-1, 1),
}

# test data contains a list of samples, the corresponding expected labels, and expected values for log likelihoods
# type: ignore
test_data_dict: dict[str, list] = {
    "samples": [
        ["foo", "bar", "love"],
        ["foo", "bar", "hate"],
        ["foo", "bar", "baz"],
        ["happy", "love", "am"],
        ["sad", "hate", "am"],
        ["happy", "love", "sad"],
        ["happy", "hate", "sad"],
        ["the", "am", "weather"],
    ],
    "labels": [1, 0, 0, 1, 0, 1, 0, 0],
    "expected": [
        np.log(0.1 / 0.05),
        np.log(0.05 / 0.1),
        0.0,
        (np.log(0.15 / 0.05) + np.log(0.1 / 0.05) + 0.0),
        (np.log(0.05 / 0.15) + np.log(0.05 / 0.1) + 0.0),
        (np.log(0.15 / 0.05) + np.log(0.1 / 0.05) + np.log(0.05 / 0.15)),
        (np.log(0.15 / 0.05) + np.log(0.05 / 0.1) + np.log(0.05 / 0.15)),
        0.0,
    ],
}


@pytest.fixture
def train_samples():
    return train_data["samples"]


@pytest.fixture
def train_samples_labels():
    return train_data["labels"]


@pytest.fixture
def test_samples():
    return test_data_dict["samples"]


@pytest.fixture
def test_samples_labels():
    return test_data_dict["labels"]


@pytest.fixture
def test_samples_expected():
    return test_data_dict["expected"]


@pytest.fixture
def trained_model(train_samples, train_samples_labels):
    model._train_naive_bayes(train_samples, train_samples_labels)
    return model


@pytest.fixture
def trained_frequencies(train_samples, train_samples_labels):
    model._get_word_frequencies(train_samples, train_samples_labels)
    return model.df_freqs


@pytest.fixture
def trained_likelihoods(train_samples, train_samples_labels):
    model._get_word_frequencies(train_samples, train_samples_labels)
    model._get_word_probabilities()
    return model.word_probabilities


@pytest.fixture
def trained_log_ratios(train_samples, train_samples_labels):
    model._get_word_frequencies(train_samples, train_samples_labels)
    model._get_word_probabilities()
    model._get_log_ratios()
    return model.log_ratios


@pytest.mark.parametrize(
    "labels, expected",
    [
        (np.array([1, 1, 1, 1, 0, 0, 0, 0]), 0.0),
        (np.array([1, 1, 1, 1, 1, 1, 0, 0]), np.log(6) - np.log(2)),
        (np.array([0, 0, 0, 0, 0, 0, 1, 1]), np.log(2) - np.log(6)),
    ],
)
def test_set_logprior(labels, expected):
    model.logprior = labels
    np.testing.assert_allclose(model.logprior, expected)
    # assert model.logprior == expected


@pytest.mark.parametrize(
    "labels, expectation",
    [
        (np.array([0, 0, 0, 0, 0, 0, 0, 0]), pytest.raises(AssertionError)),
        (np.array([1, 1, 1, 1, 1, 1, 1, 1]), pytest.raises(AssertionError)),
        (np.array([1, 1, 1, 1, 0, 0, 0, 0]), does_not_raise()),
    ],
)
def test_set_logprior_exception(labels, expectation):
    with expectation:
        model.logprior = labels


def test_get_word_frequencies(trained_frequencies):
    assert trained_frequencies.index.size == 9
    assert trained_frequencies.loc["happy", 1] == 2
    assert trained_frequencies.loc["happy", 0] == 0
    assert trained_frequencies.loc["sad", 1] == 0
    assert trained_frequencies.loc["sad", 0] == 2
    assert trained_frequencies.loc["weather", 1] == 1
    assert trained_frequencies.loc["weather", 0] == 1
    assert trained_frequencies.loc["love", 1] == 1
    assert trained_frequencies.loc["love", 0] == 0
    assert trained_frequencies.loc["hate", 1] == 0
    assert trained_frequencies.loc["hate", 0] == 1
    assert trained_frequencies.loc["I", 1] == 3
    assert trained_frequencies.loc["I", 0] == 3
    assert trained_frequencies.loc["am", 1] == 2
    assert trained_frequencies.loc["am", 0] == 2
    assert trained_frequencies.loc["because", 1] == 1
    assert trained_frequencies.loc["because", 0] == 1
    assert trained_frequencies.loc["the", 1] == 1
    assert trained_frequencies.loc["the", 0] == 1


def test_get_word_probabilities(trained_likelihoods):
    assert trained_likelihoods.index.size == 9
    assert trained_likelihoods.loc["happy", 1] == 0.15
    assert trained_likelihoods.loc["happy", 0] == 0.05
    assert trained_likelihoods.loc["sad", 1] == 0.05
    assert trained_likelihoods.loc["sad", 0] == 0.15
    assert trained_likelihoods.loc["weather", 1] == 0.1
    assert trained_likelihoods.loc["weather", 0] == 0.1
    assert trained_likelihoods.loc["love", 1] == 0.1
    assert trained_likelihoods.loc["love", 0] == 0.05
    assert trained_likelihoods.loc["hate", 1] == 0.05
    assert trained_likelihoods.loc["hate", 0] == 0.1
    assert trained_likelihoods.loc["I", 1] == 0.2
    assert trained_likelihoods.loc["I", 0] == 0.2
    assert trained_likelihoods.loc["am", 1] == 0.15
    assert trained_likelihoods.loc["am", 0] == 0.15
    assert trained_likelihoods.loc["because", 1] == 0.1
    assert trained_likelihoods.loc["because", 0] == 0.1
    assert trained_likelihoods.loc["the", 1] == 0.1
    assert trained_likelihoods.loc["the", 0] == 0.1


def test_get_log_ratios(trained_log_ratios):
    assert isinstance(trained_log_ratios, pd.Series)
    assert trained_log_ratios.index.size == 9

    np.testing.assert_allclose(trained_log_ratios.loc["happy"], np.log(0.15 / 0.05))
    np.testing.assert_allclose(trained_log_ratios.loc["sad"], np.log(0.05 / 0.15))
    np.testing.assert_allclose(trained_log_ratios.loc["love"], np.log(0.1 / 0.05))
    np.testing.assert_allclose(trained_log_ratios.loc["hate"], np.log(0.05 / 0.1))
    np.testing.assert_allclose(trained_log_ratios.loc["weather"], 0.0)
    np.testing.assert_allclose(trained_log_ratios.loc["I"], 0.0)
    np.testing.assert_allclose(trained_log_ratios.loc["am"], 0.0)
    np.testing.assert_allclose(trained_log_ratios.loc["because"], 0.0)
    np.testing.assert_allclose(trained_log_ratios.loc["the"], 0.0)


@pytest.mark.parametrize(
    "X, y, expectation",
    [
        (
            [["I", "am", "happy"], ["I", "am", "sad"], ["NLP", "is", "fun"]],
            np.array([1, 0, 1, 0]).reshape(-1, 1),
            pytest.raises(AssertionError),
        ),
        (
            [["I", "am", "happy"], ["I", "am", "sad"], ["NLP", "is", "fun"]],
            np.array([1, 0, 1]),
            pytest.raises(AssertionError),
        ),
        (
            [["I", "am", "happy"], ["I", "am", "sad"], ["NLP", "is", "fun"]],
            np.array([1, 0, 1]).reshape(1, -1),
            pytest.raises(AssertionError),
        ),
        (
            [["I", "am", "happy"], ["I", "am", "sad"], ["NLP", "is", "fun"]],
            np.array([1, 0, 1]).reshape(-1, 1),
            does_not_raise(),
        ),
    ],
)
def test_fit(X, y, expectation):
    with expectation:
        model = NaiveBayes()
        model.fit(X, y)


def test_train_naive_bayes(trained_model):
    assert trained_model.logprior == 0.0

    assert trained_model.df_freqs.index.size == 9
    assert trained_model.df_freqs.columns.size == 2
    assert trained_model.df_freqs.select_dtypes(include=["int64"]).columns.size == 2

    assert trained_model.word_probabilities.index.size == 9
    assert trained_model.word_probabilities.columns.size == 2
    assert (
        trained_model.word_probabilities.select_dtypes(include=["float64"]).columns.size
        == 2
    )


@pytest.mark.parametrize(
    "test_sample, expected",
    [pair for pair in zip(test_data_dict["samples"], test_data_dict["expected"])],
)
def test_predict_single(trained_model, test_sample, expected):
    y_pred = trained_model.predict_single(test_sample)

    np.testing.assert_allclose(y_pred, expected)


def test_predict_prob(trained_model, test_samples, test_samples_expected):
    y_pred = trained_model.predict_prob(test_samples)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (8, 1)
    np.testing.assert_allclose(y_pred, np.array(test_samples_expected).reshape(-1, 1))


def test_predict(trained_model, test_samples, test_samples_labels):
    y_pred = trained_model.predict(test_samples)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (8, 1)
    np.testing.assert_array_equal(y_pred, np.array(test_samples_labels).reshape(-1, 1))
