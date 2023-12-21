"""Tests for the logistic regression module.

"""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from htwgnlp.logistic_regression import LogisticRegression

clf = LogisticRegression()


@pytest.mark.parametrize(
    "input, expected",
    [
        (1, np.array([[0]])),
        (3, np.array([[0], [0], [0]])),
        (5, np.array([[0], [0], [0], [0], [0]])),
    ],
)
def test_initialize_weights(input, expected):
    clf._initialize_weights(input)
    np.testing.assert_allclose(clf.theta, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 0.5),
        (0.5, 0.6224593312018546),
        (-0.5, 0.3775406687981454),
        (1, 0.7310585786300049),
        (-1, 0.2689414213699951),
        (10, 0.9999546021312976),
        (-10, 4.5397868702434395e-05),
        (np.array([1, 2, 3]), np.array([0.73105858, 0.88079708, 0.95257413])),
        (
            np.array([1, 2, 3]).reshape(3, 1),
            np.array([[0.73105858], [0.88079708], [0.95257413]]),
        ),
    ],
)
def test_sigmoid(input, expected):
    np.testing.assert_allclose(clf._sigmoid(input), expected)


@pytest.mark.parametrize(
    "seed, expected",
    [
        (42, np.array([[1.23766554]])),
        (17, np.array([[1.32527587]])),
        (77, np.array([[0.79960487]])),
    ],
)
def test_cost_function(seed, expected):
    np.random.seed(seed)

    h = np.random.rand(10, 1)
    y = np.random.choice([0, 1], size=(10, 1)).astype(float)

    model = LogisticRegression()
    model._cost_function(y, h)

    np.testing.assert_allclose(model.cost, expected)


@pytest.mark.parametrize(
    "seed, expected",
    [
        (42, np.array([[1.99140094e-10], [2.18749628e-10], [1.14502157e-10]])),
        (17, np.array([[2.12509915e-10], [7.54512573e-11], [1.02384966e-10]])),
        (77, np.array([[1.52127429e-10], [1.70980203e-10], [1.44989518e-10]])),
    ],
)
def test_update_weights(seed, expected):
    np.random.seed(seed)

    h = np.random.rand(10, 1)
    y = np.random.choice([0, 1], size=(10, 1)).astype(float)
    X = np.random.rand(10, 3)

    model = LogisticRegression()
    model._initialize_weights(X.shape[1])
    model._update_weights(X, y, h)

    np.testing.assert_allclose(model.theta, expected)


@pytest.mark.parametrize(
    "seed, expected_cost, expected_theta",
    [
        (
            42,
            np.array([[0.67968487]]),
            np.array([[-0.05434972], [-0.05869232], [-0.09179451]]),
        ),
        (
            17,
            np.array([[0.68170141]]),
            np.array([[-0.0004029], [0.03615113], [0.1063462]]),
        ),
        (
            77,
            np.array([[0.69164359]]),
            np.array([[0.02648923], [0.03031277], [-0.00662695]]),
        ),
    ],
)
def test_gradient_descent(seed, expected_cost, expected_theta):
    np.random.seed(seed)

    y = np.random.choice([0, 1], size=(10, 1)).astype(float)
    X = np.random.rand(10, 3)

    model = LogisticRegression(learning_rate=0.1, n_iter=10)
    model._gradient_descent(X, y)

    np.testing.assert_allclose(model.cost, expected_cost)
    np.testing.assert_allclose(model.theta, expected_theta, rtol=1e-5)


@pytest.mark.parametrize(
    "X, y, expectation",
    [
        (
            np.random.rand(9, 3),
            np.random.choice([0, 1], size=(10, 1)).astype(float),
            pytest.raises(AssertionError),
        ),
        (
            np.random.rand(10, 3, 2),
            np.random.choice([0, 1], size=(10, 1)).astype(float),
            pytest.raises(AssertionError),
        ),
        (
            np.random.rand(10, 3),
            np.random.choice([0, 1], size=(10,)).astype(float),
            pytest.raises(AssertionError),
        ),
        (
            np.random.rand(10, 3),
            np.random.choice([0, 1], size=(10, 2)).astype(float),
            pytest.raises(AssertionError),
        ),
        (
            np.random.rand(10, 3),
            np.random.choice([0, 1], size=(10, 1)).astype(float),
            does_not_raise(),
        ),
    ],
)
def test_fit(X, y, expectation):
    with expectation:
        model = LogisticRegression(learning_rate=0.1, n_iter=10)
        model.fit(X, y)


@pytest.mark.parametrize(
    "seed, expected",
    [
        (42, np.array([[0.61933764], [0.78334267], [0.73002308]])),
        (17, np.array([[0.63720961], [0.62263409], [0.64992326]])),
        (77, np.array([[0.68525932], [0.69599701], [0.78542849]])),
    ],
)
def test_predict_prob(seed, expected):
    np.random.seed(seed)

    theta = np.random.rand(3, 1)
    X = np.random.rand(3, 3)

    model = LogisticRegression()
    model.theta = theta
    y_prob = model.predict_prob(X)

    np.testing.assert_allclose(y_prob, expected)


@pytest.mark.parametrize(
    "seed, expected",
    [
        (42, np.array([[1], [1], [1]])),
        (17, np.array([[0], [0], [0]])),
        (77, np.array([[1], [0], [1]])),
    ],
)
def test_predict(seed, expected):
    np.random.seed(seed)

    theta = np.random.randn(3, 1)
    X = np.random.rand(3, 3)

    model = LogisticRegression()
    model.theta = theta
    y_hat = model.predict(X)

    np.testing.assert_allclose(y_hat, expected)
