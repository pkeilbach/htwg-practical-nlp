# Minimum Edit Distance

Minimum edit distance is a technique that can be used to find strings that are close to a given string.

It is used in autocorrection to find words that are close to the incorrect word.

You've probably seen autocorrect in action before. It is a feature that automatically corrects your spelling mistakes as you type and is integrated in many applications and systems we use on a daily basis.

It does not always work as intended, and sometimes it can even be quite funny. You have probably seen examples like this one ...

![A funny autocorrect example](https://img.buzzfeed.com/buzzfeed-static/static/2019-10/15/16/asset/6112cf8bfdeb/sub-buzz-214-1571155829-1.jpg?downsize=600:*&output-format=auto&output-quality=auto)

... or this one ...

![Another funny autocorrect example](https://i.chzbgr.com/full/9638603264/h398BF81A/can-go-gym-tomorrow-sure-abby-baby-oh-boy-here-go-read-859-pm-imessage)

!!! tip

    Find more funny autocorrect examples [here](https://www.buzzfeed.com/andrewziegler/autocorrect-fails-of-the-decade) and [here](https://cheezburger.com/15340293/25-outrageous-autocorrect-fails-to-make-you-lol-cheezburger).

## Process

1. Identify an incorrect word
2. Find strings $n$ edit distance away
3. Filter candidate words
4. Calculate the word probabilities

!!! example

    Consider the following sentence:

    > my *deah* friend ❌

    We can see that the word `deah` is incorrect.

    > my *dear* friend ✅

    We can find words that are 1 edit distance away from `deah`:

    - dea**r**
    - dea**d**
    - dea**l**
    - **y**eah,

    Words that are two edit distances away from `deah`:

    - de**er**
    - de**ed**

    So intuitively, the edit distance tells us **how many edits** we need to make to a word to get to another word.

    With an edit distance of 4, we can change the word `deah` completely.

!!! quote

    We want to find strings that are *n edit distance* away from a given string.

## Identifying Incorrect Words

The simplified assumption we can make here is that if a word is not in the vocabulary, then it is probably a typo.

```python hl_lines="4"
def identify_incorrect_words(words, vocab):
    incorrect_words = []
    for word in words:
        if word not in vocab:
            incorrect_words.append(word)
    return incorrect_words
```

Note that the vocabulary is a **set of words**, i.e. it does not contain duplicates.

```python
>>> words = ["happy", "sad", "excited", "happy", "sad", "happy", "cool", "funny"]
>>> vocab = set(words)
>>> vocab
{'happy', 'sad', 'excited', 'funny', 'cool'}
```

!!! info

    There are much more sophisticated techniques for identifying words that are probably incorrect by looking at the context of the word.
    However, in this lecture, we will only look at spelling errors, not grammatical or contextual errors.

!!! example

    Consider the following sentence:

    - Happy birthday to you, my dear friend ✅
    - Happy birthday to you, my deah friend ❌

    This is as expected, since `deah` is not in the vocabulary. But since we can expect the word `deer` to be in the vocabulary, we would not identify the following sentence as incorrect:

    - Happy birthday to you, my deer friend ✅🦌

## Finding Strings

An edit is a type of operation that we can perform on a string to change it into another string.

An edit can be one of the following:

- Insertion
- Deletion
- Switch
- Replacement

The edit distance counts the number of operations that is needed to change one string into another string. We can combine these operations in any order.

By combining these edits, we can find a list of all possible strings that are $n$ edit distances away from a given string, regardless of wheter those strings are valid words or not.

So the edit distance $n$ tells us, how many operations one string is away from another string.

!!! info

    For autocorrect, we want to find strings that are close to the incorrect word. In our lecture, this means we want to find strings that are not more than 3 edit distances away from the incorrect word.

### Insertion

An insertion is when we insert a character into a string. It does not matter where we insert the character.

```python
def insert_char(word, i, char):
    return word[:i] + char + word[i:]
```

!!! example

    For the word `to`

    - if we do an insertion of the character `p`, we get the word `top`.
    - if we do an insertion of the character `o`, we get the word `too`.
    - if we do an insertion of the character `w`, we get the word `two`.

### Deletion

A deletion is when we remove a character from a string. It does not matter which character we delete.

```python
def delete_char(word, i):
    return word[:i] + word[i+1:]
```

!!! example

    For the word `hat`

    - if we remove the character `h`, we get the word `at`.
    - if we remove the character `a`, we get the word `ht`.
    - if we remove the character `t`, we get the word `ha`.

### Switch

A switch operation is when we swap two characters in a string that are **next to each other** (adjacent).

```python
def switch_char(word, i):
    return word[:i] + word[i+1] + word[i] + word[i+2:]
```

!!! example

    For the word `eta`

    - if we swap the characters `t` and `a`, we get the word `eat`.
    - if we swap the characters `t` and `e`, we get the word `tea`.

    But we cannot swap the characters `e` and `a` to get the word `ate`, because the characters are not next to each other.

### Replacement

A replacement is when we replace a character in a string with another character. It does not matter which character we replace.

```python
def replace_char(word, i, char):
    return word[:i] + char + word[i+1:]
```

!!! example

    For the word `jaw`

    - if we replace the character `j` with `s`, we get the word `saw`.
    - if we replace the character `w` with `r`, we get the word `jar`.
    - if we replace the character `j` with `p`, we get the word `paw`.

## Filtering Candidate Words

Many words returned by the edit distance algorithm are not valid words. We need to filter these words out.

To do this, we can check if the word is in the vocabulary. If it is, then we can keep it.

```python hl_lines="4"
def filter_candidates(words, vocab):
    valid_words = []
    for word in words:
        if word in vocab:
            valid_words.append(word)
    return valid_words
```

## Calculating Word Probabilities

If we have the candidate words available, the next step is to find the word that is most likely to be the correct word (or the $n$ most likely words).

!!! example

    Consider the following sentence:

    "I like apples *ans* bananas"

    The word `ans` is supposedly a typo of the word `and`.

    If we look at two candidate words, `and` and `ant`, we can observe that both are 1 edit distance away from `ans`. So how can we tell that `and` is more likely to be the correct word than `ant`? 🐜

    Usually, we can assume that the word `and` is more frequent than the word `ant` 🐜 in any given text, so based on word frequencies, the model should suggest the word `and` as the correct word. 💡

To do this, we can use the word probabilities. The word probabilities tell us how likely it is for a word to appear in a given text.

The word probabilities can be calculated by counting the number of times a word appears in a text, and dividing it by the total number of words in the text.

$$
P(w) = \frac{N(w)}{N}
$$

Where

- $P(w)$ is the probability of the word $w$ appearing in a text
- $N(w)$ is the frequency of the word $w$ in the text
- $N$ is the total count of all words in the text

!!! info

    This is quite similar as from the lecture on [feature extraction](./feature_extraction.md#positive-and-negative-frequencies), where we calculated the word frequencies per class. The difference here is that we are now calculating the word frequencies for the entire text, not per class.

!!! example

    Consider the following corpus:

    ```python
    corpus = [
        "I like apples and bananas",
        "I like apples and oranges"
    ]
    ```

    By counting the number of times each word appears in the corpus, we get the word frequencies. Since the total number of words $N$ in the corpus is 10, we can calculate the word probabilities, and end up with the following table:

    If we build the word frequency table for this corpus, we get the following:

    | Word     | Frequency | Probability |
    | -------- | --------- | ----------- |
    | I        | 2         | 0.2         |
    | like     | 2         | 0.2         |
    | apples   | 2         | 0.2         |
    | and      | 2         | 0.2         |
    | bananas  | 1         | 0.1         |
    | oranges  | 1         | 0.1         |
    | **total**    | **10**        | **1.0**        |

Note that in Python, we can utilize the `Counter` class from the `collections` module to count the number of times each word appears in the corpus.

```python
>>> from collections import Counter
>>> freqs = Counter()
>>> for text in corpus:
>>>     freqs.update(text.split())
>>> freqs
Counter({'I': 2, 'like': 2, 'apples': 2, 'and': 2, 'bananas': 1, 'oranges': 1})
```

From there, we can calculate the word probabilities as follows:

```python
>>> total_words = sum(freqs.values())
>>> {word: freq / total_words for word, freq in freqs.items()}
{'I': 0.2, 'like': 0.2, 'apples': 0.2, 'and': 0.2, 'bananas': 0.1, 'oranges': 0.1}
```

!!! info

    The code snippet above shows a Python [dict comprehension](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).

    They can be used to create dictionaries from arbitrary key and value expressions, like so:

    ```python
    >>> {x: x**2 for x in (2, 4, 6)}
    {2: 4, 4: 16, 6: 36}
    ```

    This is equivalent to the following code:

    ```python
    data = {}
    for x in (2, 4, 6):
        data[x] = x**2
    ```

## Limitations

> my dear/deer friend
