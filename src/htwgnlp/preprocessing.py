"""Tweet preprocessing module.

This module contains the TweetProcessor class which is used to preprocess tweets.

ASSIGNMENT-1:
Your job in this assignment is to implement the methods of this class.
Note that you will need to import several modules from the nltk library, 
as well as from the Python standard library.
You can find the documentation for the nltk library here: https://www.nltk.org/
You can find the documentation for the Python standard library here: https://docs.python.org/3/library/
Your task is complete when all the tests in the test_preprocessing.py file pass.
You can check if the tests pass by running `make assignment_1` in the terminal.
You can follow the `TODO ASSIGNMENT-1` comments to find the places where you need to implement something.
"""


from nltk.stem import PorterStemmer
import re  # used for pattern matching and manipulation of strings.
import string


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
        tweet = re.sub(r"http\S+", "", tweet)
        return tweet
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
        tweet = re.sub(r"#", "", tweet)
        return tweet
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
        # 1) Lowercase the text
        tweet = tweet.lower()

        # 2) Identify URLs in the tweet and emojis in the tweet . for example: ( :) , :D, :P, :/ , :| , :O , :S , :@ , :$ , :* , :3 , :') , :( )
        urls = re.findall(r"https?://\S+", tweet)
        for url in urls:
            tweet = tweet.replace(url, f"__url_placeholder__")

        # Identify Emojis in the tweet
        emoji_pattern = re.compile(
            r"(:\)|:-\)|:\(|:-\(|:D|:d|:P|:p|:/|:\||:O|:S|:@|:\$|:\*|:3|:\'\()"
        )
        # Find all matches in the tweet
        emojis = re.findall(emoji_pattern, tweet)
        for emoji in emojis:
            tweet = tweet.replace(emoji, f"__emoji_placeholder__")

        # 3) Separate specified punctuation from words using regular expression
        tweet = re.sub(r"([.,;:!?])", r" \1 ", tweet)

        # 4) Remove handles (assuming handles start with '@')
        tweet = re.sub(r"@\S+", "", tweet)

        # 5) Reduce repeated characters to a maximum of two
        tweet = re.sub(r"(.)\1{3,}", r"\1\1\1", tweet)

        # 6) Replace Url placeholder with original url and Replace Emoji placeholder with original emoji
        for url in urls:
            tweet = tweet.replace("__url_placeholder__", url)
        for emoji in emojis:
            tweet = tweet.replace("__emoji_placeholder__", emoji)

        # 7) Split the tweet into tokens (assuming words are separated by spaces)
        tokens = tweet.split()
        return tokens

        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        """Removes stopwords from a tweet.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without stopwords
        """

        stopwords = [
            "i",
            "me",
            "my",
            "myself",
            "be",
            "we",
            "was",
            "because",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "the",
            "for",
            "is",
            "just",
            "can",
            "to",
            "out",
            "of",
            "it",
            "a",
            "it's",
            "very",
            "with",
            "too",
            "and",
            "that",
            "this",
            "have",
            "in",
            "on",
            "at",
            "as",
            "so",
            "but",
            "if",
            "or",
            "by",
            "from",
            "up",
            "down",
            "about",
            "into",
            "over",
            "after",
            "then",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "are",
        ]
        tokens = [word for word in tokens if word not in stopwords]
        return tokens

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

        # Define a set of punctuation characters
        punctuation_set = set(char for char in string.punctuation if char != "'")

        # Use list comprehension to filter out tokens that are not punctuation
        filtered_tokens = [token for token in tokens if token not in punctuation_set]

        return filtered_tokens

        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    def stem(self, tokens: list[str]) -> list[str]:
        """Stems the tokens of a tweet using the nltk PorterStemmer.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet with stemmed tokens
        """

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens

        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")

    def process_tweet(self, tweet: str) -> list[str]:
        """Processes a tweet by removing urls, hashtags, stopwords, punctuation, and stemming the tokens.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the processed tweet
        """

        # Remove URLs
        tweet = self.remove_urls(tweet)

        # Remove hashtags
        tweet = self.remove_hashtags(tweet)

        # Tokenize the tweet
        tokens = self.tokenize(tweet)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Remove punctuation
        tokens = self.remove_punctuation(tokens)

        # Stem the tokens
        tokens = self.stem(tokens)

        return tokens

        # TODO ASSIGNMENT-1: implement this function
        raise NotImplementedError("This function needs to be implemented.")
