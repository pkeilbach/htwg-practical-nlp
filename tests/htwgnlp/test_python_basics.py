"""Tests for `python_basics.py` module using pytest.

This test file can be run using `make assignment-0`
"""

import os
from collections import Counter

import pytest

from htwgnlp.python_basics import (
    build_phrases,
    count_purchases,
    get_even_numbers,
    get_long_words,
    get_top_x_products,
    get_uppercase_words,
    get_word_lengths,
    print_product_price,
    read_dict_from_json_file,
    slicing_examples,
    sort_people_by_age,
    write_dict_to_json_file,
)


@pytest.mark.parametrize(
    "numbers, expected_output",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 2, 4, 6, 8, 10]),
        ([], []),
        ([-10, -5, 0, 5, 10], [-10, 0, 10]),
        ([1, 3, 5, 7], []),
        ([2, 4, 6, 8], [2, 4, 6, 8]),
    ],
)
def test_get_even_numbers(numbers, expected_output):
    assert get_even_numbers(numbers) == expected_output


@pytest.mark.parametrize(
    "words, expected_output",
    [
        (
            ["apple", "banana", "cherry", "elderberry", "mango", "fig"],
            ["banana", "cherry", "elderberry"],
        ),
        ([], []),
        (["short", "tiny", "small"], []),
    ],
)
def test_get_long_words(words, expected_output):
    assert get_long_words(words) == expected_output


@pytest.mark.parametrize(
    "words, expected_output",
    [
        (
            ["apple", "banana", "cherry", "elderberry", "mango", "fig"],
            ["APPLE", "BANANA", "CHERRY", "ELDERBERRY", "MANGO", "FIG"],
        ),
        ([], []),
        (["a", "b", "c"], ["A", "B", "C"]),
        (["UPPER", "lower", "Mixed"], ["UPPER", "LOWER", "MIXED"]),
        (["123", "456", "789"], ["123", "456", "789"]),
    ],
)
def test_get_uppercase_words(words, expected_output):
    assert get_uppercase_words(words) == expected_output


@pytest.mark.parametrize(
    "adjectives, animals, expected_output",
    [
        (
            ["big", "small", "furry", ""],
            ["cat", "dog", "rabbit", ""],
            [
                "big cat",
                "big dog",
                "big rabbit",
                "small cat",
                "small dog",
                "small rabbit",
                "furry cat",
                "furry dog",
                "furry rabbit",
            ],
        ),
        ([], ["cat", "dog", "rabbit"], []),
        (["big", "small", "furry"], [], []),
        ([], [], []),
    ],
)
def test_build_phrases(adjectives, animals, expected_output):
    assert build_phrases(adjectives, animals) == expected_output


@pytest.mark.parametrize(
    "words, expected_output",
    [
        (
            ["apple", "banana", "cherry", "dates", "elderberry", ""],
            {"apple": 5, "banana": 6, "cherry": 6, "dates": 5, "elderberry": 10, "": 0},
        ),
        ([], {}),
    ],
)
def test_get_word_lengths(words, expected_output):
    assert get_word_lengths(words) == expected_output


@pytest.mark.parametrize(
    "product, price, expected_output",
    [
        ("banana", 1.5, 'The price of the product "banana" is 1.50 USD.'),
        ("apple", 2, 'The price of the product "apple" is 2.00 USD.'),
        ("orange", 0.99, 'The price of the product "orange" is 0.99 USD.'),
        ("grape", 10.123, 'The price of the product "grape" is 10.12 USD.'),
        ("watermelon", 5.6789, 'The price of the product "watermelon" is 5.68 USD.'),
    ],
)
def test_print_product_price(product, price, expected_output):
    assert print_product_price(product, price) == expected_output


@pytest.mark.parametrize(
    "product, price",
    [
        ("banana", -1.5),
        ("apple", 0),
        ("orange", -0.99),
    ],
)
def test_print_product_price_value_error(product, price):
    with pytest.raises(ValueError):
        print_product_price(product, price)


@pytest.mark.parametrize(
    "purchases, expected_output",
    [
        (
            ["apple", "banana", "apple", "orange", "banana", "apple"],
            Counter({"apple": 3, "banana": 2, "orange": 1}),
        ),
        ([], Counter()),
    ],
)
def test_count_purchases(purchases, expected_output):
    assert count_purchases(purchases) == expected_output


def test_get_top_products():
    purchases = [
        "apple",
        "banana",
        "apple",
        "orange",
        "banana",
        "apple",
        "mandarin",
        "banana",
        "apple",
        "orange",
        "banana",
        "fig",
        "apple",
        "orange",
        "banana",
        "fig",
        "apple",
        "orange",
    ]
    assert get_top_x_products(purchases, 3) == [
        ("apple", 6),
        ("banana", 5),
        ("orange", 4),
    ]
    assert get_top_x_products(purchases, 2) == [("apple", 6), ("banana", 5)]
    assert get_top_x_products(purchases, 0) == []


@pytest.mark.parametrize(
    "people, expected_output",
    [
        (
            [("Alice", 30), ("Bob", 25), ("Charlie", 35)],
            [("Bob", 25), ("Alice", 30), ("Charlie", 35)],
        ),
        ([], []),
        (
            [("Bob", 30), ("Charlie", 30), ("Alice", 30)],
            [("Alice", 30), ("Bob", 30), ("Charlie", 30)],
        ),
    ],
)
def test_sort_people_by_age(people, expected_output):
    assert sort_people_by_age(people) == expected_output


def test_write_dict_to_json_file():
    data = {"name": "Alice", "age": 30, "city": "New York"}
    filename = "tests/htwgnlp/files/test_output.json"
    write_dict_to_json_file(data, filename)
    assert os.path.exists(filename)
    os.remove(filename)


def test_read_dict_from_json_file():
    expected_data = {"name": "Alice", "age": 30, "city": "New York"}
    filename = "tests/htwgnlp/files/test_data.json"
    data = read_dict_from_json_file(filename)

    assert data == expected_data


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        (
            [1, 2, 3, 4, 5, 6],
            {
                "first_three": [1, 2, 3],
                "last_two": [5, 6],
                "reversed": [6, 5, 4, 3, 2, 1],
                "skip_two": [1, 3, 5],
                "middle_slice": [2, 3, 4, 5],
            },
        ),
        (
            [10, 20, 30],
            {
                "first_three": [10, 20, 30],
                "last_two": [20, 30],
                "reversed": [30, 20, 10],
                "skip_two": [10, 30],
                "middle_slice": [20],
            },
        ),
        (
            [1],
            {
                "first_three": [1],
                "last_two": [1],
                "reversed": [1],
                "skip_two": [1],
                "middle_slice": [],
            },
        ),
        (
            [],
            {
                "first_three": [],
                "last_two": [],
                "reversed": [],
                "skip_two": [],
                "middle_slice": [],
            },
        ),
    ],
)
def test_slicing_examples(input_list, expected_output):
    assert slicing_examples(input_list) == expected_output
