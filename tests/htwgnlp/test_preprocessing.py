"""Tests for the preprocessing module.

This module contains the tests for the TweetProcessor class.
Run the tests with `make assignment-1` in the terminal.
You don't need to worry about the implementation of this module or the pytest framework,
but if you are wondering what pytest.mark.parametrize does, check out the pytest documentation here:
https://docs.pytest.org/en/stable/parametrize.html
"""

import pytest

from htwgnlp.preprocessing import TweetProcessor

processor = TweetProcessor()


@pytest.mark.parametrize(
    "input_tweet, expected_result",
    [
        (
            "Check out this cool website: https://example.com",
            "Check out this cool website: ",
        ),
        (
            "Just posted a new blog: https://myblog.com/how-to-do-something",
            "Just posted a new blog: ",
        ),
        ("No URLs in this tweet!", "No URLs in this tweet!"),
        (
            "Visit my GitHub profile at https://github.com/user123",
            "Visit my GitHub profile at ",
        ),
        ("https://invalid-url-without-space.com", ""),
        (
            "Multiple URLs: https://example.com and https://another.com",
            "Multiple URLs:  and ",
        ),
        (
            "URLs like this one: http://example.com and https://another.com",
            "URLs like this one:  and ",
        ),
        (
            "URLs with the http-protocol like this one: http://example.com and http://another.com",
            "URLs with the http-protocol like this one:  and ",
        ),
        ("https://example.com/with/path?query=1", ""),
    ],
)
def test_remove_urls(input_tweet, expected_result):
    assert processor.remove_urls(input_tweet) == expected_result


@pytest.mark.parametrize(
    "input_tweet, expected_result",
    [
        ("Check out #my cool #hashtag", "Check out my cool hashtag"),
        ("#SingleHashtag", "SingleHashtag"),
        ("No hashtags in this tweet!", "No hashtags in this tweet!"),
        ("#Multiple #hashtags in #one #tweet", "Multiple hashtags in one tweet"),
        ("#hashtag#attachedtogether", "hashtagattachedtogether"),
        (
            "Hashtags at the beginning #starting and ending #ending of the tweet",
            "Hashtags at the beginning starting and ending ending of the tweet",
        ),
        ("#1stHashtag with a number", "1stHashtag with a number"),
        ("#SpecialCharacters in #hashtags!", "SpecialCharacters in hashtags!"),
    ],
)
def test_remove_hashtags_from_tweet(input_tweet, expected_result):
    assert processor.remove_hashtags(input_tweet) == expected_result


@pytest.mark.parametrize(
    "input_tweet, expected_tokens",
    [
        (
            "Hello @JohnDoe, how are you today?",
            ["hello", ",", "how", "are", "you", "today", "?"],
        ),
        ("@MentionedUser thanks for the help!", ["thanks", "for", "the", "help", "!"]),
        ("I looooveee itttt!!!", ["i", "loooveee", "ittt", "!", "!", "!"]),
        ("#NLTK is awesome!", ["#nltk", "is", "awesome", "!"]),
        ("Retweet if you agree!", ["retweet", "if", "you", "agree", "!"]),
        (
            "Check out this link: https://example.com",
            ["check", "out", "this", "link", ":", "https://example.com"],
        ),
        ("Emoticons are fun 🤓 :)", ["emoticons", "are", "fun", "🤓", ":)"]),
        ("#HashtagsAreGreat #PythonIsCool", ["#hashtagsaregreat", "#pythoniscool"]),
    ],
)
def test_tokenize(input_tweet, expected_tokens):
    assert processor.tokenize(input_tweet) == expected_tokens


@pytest.mark.parametrize(
    "tokenized_tweet, expected_filtered_tweet",
    [
        # No stopwords to remove
        (["hello", "world"], ["hello", "world"]),
        (["great", "job", "team"], ["great", "job", "team"]),
        # Removing common stopwords
        (["i", "love", "nltk"], ["love", "nltk"]),
        (["the", "quick", "brown", "fox"], ["quick", "brown", "fox"]),
        (["just", "watched", "movie"], ["watched", "movie"]),
        (["#NLTK", "is", "amazing"], ["#NLTK", "amazing"]),
        (["@user", "Thanks", "for", "your", "help"], ["@user", "Thanks", "help"]),
        # Stopwords are case-sensitive, so none are removed
        (
            ["THANKS", "@official", "FOR", "retweet"],
            ["THANKS", "@official", "FOR", "retweet"],
        ),
    ],
)
def test_remove_stopwords(tokenized_tweet, expected_filtered_tweet):
    assert processor.remove_stopwords(tokenized_tweet) == expected_filtered_tweet


# test the remove punctuation function
@pytest.mark.parametrize(
    "tokenized_tweet, expected_filtered_tweet",
    [
        (["hello", "world"], ["hello", "world"]),
        (["hi", ",", "how", "are", "you", "?"], ["hi", "how", "are", "you"]),
        (
            ["@user", "Thanks", "for", "the", "help", "!"],
            ["@user", "Thanks", "for", "the", "help"],
        ),
        (["I", "love", "NLTK", "."], ["I", "love", "NLTK"]),
        (
            ["#NLTK", "is", "awesome", "!", "!", "!", "#NaturalLanguageProcessing"],
            ["#NLTK", "is", "awesome", "#NaturalLanguageProcessing"],
        ),
        (
            ["Parentheses", "(", "and", ")", "should", "go", "away", "!"],
            ["Parentheses", "and", "should", "go", "away"],
        ),
        (["Don't", "remove", "contractions", "!"], ["Don't", "remove", "contractions"]),
        (
            ["Emoticons", "are", "fun", "and", "should", "stay", ":)", ":D"],
            ["Emoticons", "are", "fun", "and", "should", "stay", ":)", ":D"],
        ),
    ],
)
def test_remove_punctuation(tokenized_tweet, expected_filtered_tweet):
    assert processor.remove_punctuation(tokenized_tweet) == expected_filtered_tweet


@pytest.mark.parametrize(
    "tokenized_tweet, expected_result",
    [
        (
            ["I", "love", "running", "in", "the", "park"],
            ["i", "love", "run", "in", "the", "park"],
        ),
        (
            ["Eating", "ice", "cream", "on", "Sundays", "is", "great"],
            ["eat", "ice", "cream", "on", "sunday", "is", "great"],
        ),
        (
            ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            ["the", "quick", "brown", "fox", "jump", "over", "the", "lazi", "dog"],
        ),
        (["Python", "programming", "is", "fun"], ["python", "program", "is", "fun"]),
        (
            ["I", "wish", "I", "could", "travel", "more"],
            ["i", "wish", "i", "could", "travel", "more"],
        ),
        (["Coding", "is", "awesome"], ["code", "is", "awesom"]),
        (
            ["The", "weather", "is", "sunny", "today"],
            ["the", "weather", "is", "sunni", "today"],
        ),
        (
            ["Reading", "books", "is", "my", "favorite", "hobby"],
            ["read", "book", "is", "my", "favorit", "hobbi"],
        ),
    ],
)
def test_stem(tokenized_tweet, expected_result):
    assert processor.stem(tokenized_tweet) == expected_result


@pytest.mark.parametrize(
    "input_tweet, expected_result",
    [
        # positive samples: 1144, 1348, 2371, 4158
        ("One word? :p https://t.co/pfxsm8w3eT", ["one", "word", ":p"]),
        (
            "Dear @SuttonObserver. Please can you ask your paperboys to keep the papers out of the rain today? It's a very special issue :-)",
            [
                "dear",
                "pleas",
                "ask",
                "paperboy",
                "keep",
                "paper",
                "rain",
                "today",
                "special",
                "issu",
                ":-)",
            ],
        ),
        (
            "@clarkkrm I'll be living with a vegetarian too so I'm sure she'll help me :)",
            ["i'll", "live", "vegetarian", "i'm", "sure", "she'll", "help", ":)"],
        ),
        (
            "@Gotzefying Im trying to D/L MH3 english patch for the psp :D",
            ["im", "tri", "l", "mh3", "english", "patch", "psp", ":d"],
        ),
        # negative samples: 56, 2231, 2941, 3038
        (
            "@itsNotMirna I was so sad because Elhaida was robbed by the juries :( she came 10th in the televoting",
            ["sad", "elhaida", "rob", "juri", ":(", "came", "10th", "televot"],
        ),
        (
            "This taxi is emptier than Ciara's concert :(",
            ["taxi", "emptier", "ciara'", "concert", ":("],
        ),
        (
            "Tbh, bestfriend breakups are even worse than relationship breakups. :(",
            [
                "tbh",
                "bestfriend",
                "breakup",
                "even",
                "wors",
                "relationship",
                "breakup",
                ":(",
            ],
        ),
        (
            "Still on the outside looking in at all fun going on with @bemeapp ... somebody please code me up :(",
            [
                "still",
                "outsid",
                "look",
                "fun",
                "go",
                "...",
                "somebodi",
                "pleas",
                "code",
                ":(",
            ],
        ),
    ],
)
def test_process_tweet(input_tweet, expected_result):
    assert processor.process_tweet(input_tweet) == expected_result
