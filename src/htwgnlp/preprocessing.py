"""Tweet preprocessing module.

This module contains the TweetProcessor class which is used to preprocess tweets.

ASSIGNMENT-1:
Your job in this assignment is to implement the methods of this class.
Note that you will need to import several modules from the nltk library, 
as well as from the Python standard library.
You can find the documentation for the nltk library here: https://www.nltk.org/
You can find the documentation for the Python standard library here: https://docs.python.org/3/library/
Your task is complete when all the tests in the test_preprocessing.py file pass.
You can check if the tests pass by running `make assignment-1` in the terminal.
You can follow the `TODO ASSIGNMENT-1` comments to find the places where you need to implement something.
"""


class TweetProcessor:
    # TODO ASSIGNMENT-1: Add a `stemmer` attribute to the class

    # TODO ASSIGNMENT-1: Add a `tokenizer` attribute to the class
    #  - text should be lowercased
    #  - handles should be stripped
    #  - the length should be reduced for repeated characters

    @staticmethod
    def remove_urls(tweet: str) -> str:
        """Remove urls from a tweet.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without urls
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    @staticmethod
    def remove_hashtags(tweet: str) -> str:
        """Remove hashtags from a tweet.
        Only the hashtag symbol is removed, the word itself is kept.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without hashtags symbols
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    def tokenize(self, tweet: str) -> list[str]:
        """Tokenizes a tweet using the nltk TweetTokenizer.
        This also lowercases the tweet, removes handles, and reduces the length of repeated characters.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the tokenized tweet
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        """Removes stopwords from a tweet.

        Only English stopwords are removed.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without stopwords
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    @staticmethod
    def remove_punctuation(tokens: list[str]) -> list[str]:
        """Removes punctuation from a tweet.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without punctuation
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    def stem(self, tokens: list[str]) -> list[str]:
        """Stems the tokens of a tweet using the nltk PorterStemmer.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet with stemmed tokens
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    def process_tweet(self, tweet: str) -> list[str]:
        """Processes a tweet by removing urls, hashtags, stopwords, punctuation, and stemming the tokens.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the processed tweet
        """
        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")
