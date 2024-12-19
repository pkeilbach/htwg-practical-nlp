"""Module for Python basics exercises.

This module contains some exercises to learn Python basics.
It covers some programming concepts and language features that will be useful for the course.
"""

import json
from collections import Counter


def get_even_numbers(numbers: list[int]) -> list[int]:
    """Returns a new list that contains only the even numbers.

    Use a list comprehension to solve this exercise.

    Args:
        numbers (list[int]): a list of numbers

    Returns:
        list[int]: a new list that contains only the even numbers

    Example:
        >>> get_even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        [2, 4, 6, 8, 10]
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def get_long_words(words: list[str]) -> list[str]:
    """Returns a new list that contains only the words that have more than 5 characters.

    Use a list comprehension to solve this exercise.

    Args:
        words (list[str]): a list of words

    Returns:
        list[str]: a new list that contains only the words that have more than 5 characters

    Example:
        >>> get_long_words(["apple", "banana", "cherry", "elderberry", "mango", "fig"])
        ["banana", "cherry", "elderberry"]
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def get_uppercase_words(words: list[str]) -> list[str]:
    """Returns a new list that contains the words in uppercase.

    Use a list comprehension to solve this exercise.

    Args:
        words (list[str]): a list of words

    Returns:
        list[str]: a new list that contains the words in uppercase

    Example:
        >>> get_uppercase_words(["apple", "banana", "cherry", "dates", "elderberry"])
        ["APPLE", "BANANA", "CHERRY", "DATES", "ELDERBERRY"]
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def build_phrases(adjectives: list[str], animals: list[str]) -> list[str]:
    """Returns a list of phrases by combining each adjective with each animal.

    This function takes two lists: one containing adjectives and the other containing animals.
    It returns a new list containing all possible combinations of adjectives and animals in the format "adjective animal".

    You should use a nested list comprehension to solve this exercise.

    Remember that you should not include empty strings in the output list.

    Args:
        adjectives (list of str): A list of adjectives.
        animals (list of str): A list of animals.

    Returns:
        list of str: A list containing all possible combinations of adjectives and animals.

    Example:
        >>> build_phrases(["big", "small", "furry", ""], ["cat", "dog", "rabbit", ""])
        ['big cat', 'big dog', 'big rabbit', 'small cat', 'small dog', 'small rabbit', 'furry cat', 'furry dog', 'furry rabbit']
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def get_word_lengths(words: list[str]) -> dict[str, int]:
    """Returns a dictionary with words as keys and their lengths as values.

    Use a dictionary comprehension to solve this exercise.

    Args:
        words (list of str): A list of words.

    Returns:
        dict: A dictionary where the keys are the words from the input list and the values are the lengths of those words.

    Example:
        >>> get_word_lengths(["apple", "banana", "cherry", "dates", "elderberry"])
        {'apple': 5, 'banana': 6, 'cherry': 6, 'dates': 5, 'elderberry': 11}
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def print_product_price(product: str, price: int | float) -> str:
    """Returns a string that states the price of a given product.

    Note that the price should be formatted with two decimal places.

    Use f-string formatting to solve this exercise.

    Args:
        product (str): The name of the product.´
        price (int or float): The price of the product. Must be a positive number.

    Returns:
        str: A formatted string stating the price of the product in USD.

    Raises:
        ValueError: If the price is not a positive number.

    Example:
        >>> print_product_price("banana", 1.5)
        'The price of the product "banana" is 1.50 USD.'
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def count_purchases(purchases: list[str]) -> Counter:
    """Count the number of times each product was purchased.

    Args:
        purchases (list): A list of strings where each string represents a product purchased by a customer.

    Returns:
        Counter: A Counter object where the keys are the products and the values are the counts of each product.

    Example:
        >>> purchases = ["apple", "banana", "apple", "orange", "banana", "apple"]
        >>> count_purchases(purchases)
        Counter({'apple': 3, 'banana': 2, 'orange': 1})
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def get_top_x_products(purchases: list[str], x: int) -> list[tuple[str, int]]:
    """Get the top 3 most popular products from a list of purchases.

    Args:
        purchases (list): A list of strings where each string represents a product purchased by a customer.
        x (int): The number of most popular products to return.

    Returns:
        list: A list of tuples where each tuple contains a product and its count,
              sorted by the most popular products in descending order.
              The list contains x tuples.

    Example:
        purchases = [
            "apple", "banana", "apple", "orange", "banana", "apple",
            "mandarin", "banana", "apple", "orange", "banana", "fig",
            "apple", "orange", "banana", "fig", "apple", "orange"
        ]
        get_top_x_products(purchases, 3)
        # Output: [('apple', 6), ('banana', 5), ('orange', 4)]
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def sort_people_by_age(people: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Sort a list of people by their age.

    If two people have the same age, they should be sorted by their name in ascending order.

    Args:
        people (list of tuple): A list of tuples where each tuple contains a person's name (str) and age (int).

    Returns:
        list of tuple: The list of people sorted by age in ascending order.

    Example:
        >>> people = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
        >>> sort_people_by_age(people)
        [("Bob", 25), ("Alice", 30), ("Charlie", 35)]
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def write_dict_to_json_file(data: dict, filename: str) -> None:
    """Write the contents of a dictionary to a file in JSON format.

    Args:
        data (dict): The dictionary to write to the file.
        filename (str): The path to the file where the JSON data will be written.

    Example:
        data = {
            "name": "Alice",
            "age": 30,
            "city": "New York"
        }
        write_dict_to_json_file(data, 'output.json')
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def read_dict_from_json_file(filename: str) -> dict:
    """Reads the contents of a JSON file and returns it as a dictionary.

    Args:
        filename (str): The path to the JSON file to be read.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")


def slicing_examples(input_list):
    """
    This function takes a list and returns a dictionary with results of different slicing operations.

    Implement the following:
    1. "first_three": Get the first three elements.
    2. "last_two": Get the last two elements.
    3. "reversed": Reverse the list.
    4. "skip_two": Get every second element in the list.
    5. "middle_slice": Get all elements except the first and last.

    Example usage:
    slicing_examples([1, 2, 3, 4, 5, 6])
    Expected output:
    {
        "first_three": [1, 2, 3],
        "last_two": [5, 6],
        "reversed": [6, 5, 4, 3, 2, 1],
        "skip_two": [1, 3, 5],
        "middle_slice": [2, 3, 4, 5]
    }

    Args:
        input_list (list): A list of elements.

    Returns:
        dict: A dictionary with keys as the operation names and values as the resulting sliced lists.
    """
    # TODO ASSIGNMENT-0: implement this function
    raise NotImplementedError("This function needs to be implemented.")
