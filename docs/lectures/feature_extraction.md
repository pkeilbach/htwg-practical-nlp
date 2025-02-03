# Feature Extraction

## Vocabulary

The first step to represent a text as a vector is to build a vocabulary.

In NLP, a **vocabulary** is a set of unique words in a corpus.

The vocabulary is build _after_ the preprocessing step.

$$ V = \{w_1, w_2, \ldots, w_n\} $$

```python
>>> corpus = [
... "This is the first document",
... "This document is the second document",
... "And this is the third one",
... "Here is yet another document"
... ]
>>> words = [word.lower() for document in corpus for word in document.split()]
>>> vocabulary = set(words)
>>> vocabulary
{"document", "this", "here", "one", "is", "yet", "another", "third", "second", "and", "the", "first"}
```

## One-Hot Encoding

In a one-hot encoded vector, each word in the vocabulary $V$ is assigned a unique index. Each element in the vector is binary, indicating the presence (1) or absence (0) of the corresponding word in the document.

The dimension of a feature vector $x$ is equal to the size of the vocabulary $|V|$:

$$ dim(x) = |V| $$

Here is a coding example:

```python
>>> vocabulary = {"document", "this", "here", "one", "is", "yet", "another", "third", "second", "and", "the", "first"}
>>> document = ["this", "document", "is", "the", "second", "cool", "document"]
>>> np.array([1 if word in document else 0 for word in vocabulary])
array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
```

!!! info "Sparse Representation"

    This type of representation of a vector is also known as **sparse representation**.
    In sparse representation, only a small number of elements in the representation have non-zero values, while the rest are explicitly or implicitly considered zero.

!!! warning "Limitations"

    One-hot encoding does not capture the frequency, or more complex properties like grammar or word order. It only indicates whether a word is present or not.

## Bag of Words

Bag of Words (BoW) is a similar representation to one-hot encoding.
The difference is that BoW captures the frequency of words in a document, which is a more informative representation than one-hot encoding.

```python
>>> vocabulary = {"document", "this", "here", "one", "is", "yet", "another", "third", "second", "and", "the", "first"}
>>> document = ["this", "document", "is", "the", "second", "cool", "document"]
>>> np.array([document.count(word) for word in vocabulary])
array([2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
```

!!! info

    In BoW, the values in the vector can be integers representing word counts or real numbers representing [TF-IDF weights](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

!!! warning "Dimensionality"

    With sparse representation, because of $dim(x) = |V|$, as the vocabulary size increases, the number of features increases as well.
    For the model, this means we have to fit more parameters, which negatively affects the model's performance.

## Positive and Negative Frequencies

To avoid the problem of dimensionality, we can simplify the feature vector by using only the **frequencies per class**.

Let's assume we have a binary classification problem with two classes: positive and negative.

When using only the positive and negative frequencies as features, we will only have **three features**:

- The bias unit
- The sum of positive frequencies of the words in the document
- The sum of negative frequencies of the words in the document

Compared to a sparse representation, this will significantly reduce the number of features and thus, improve the model's performance, because we have fewer parameters to fit.

!!! info "Bias Unit"

    The **bias unit** (also known as the intercept or offset) in logistic regression serves to shift the decision boundary away from the origin (0,0) in the feature space.

    Without the bias term, the decision boundary would always pass through the origin, but this is barely the case in real-world data.

    While the weights associated with the features determine the slope of the decision boundary, the bias term influences where the decision boundary is positioned on the y-axis.

!!! note

    Note that this **simplification** also means that we **lose some information**, since we are not considering the frequency of each word in the document, but only the sum of positive and negative frequencies.

Here is an example corpus and its corresponding labels:

```python
corpus = [
    "I am happy",
    "I am sad",
    "I am happy because I love the weather",
    "I am sad because I hate the weather",
]

labels = [1, 0, 1, 0]
```

We want to build a data structure that shows us the **word frequencies per class**.

- To get the positive frequency of any word in the vocabulary, we have to count how often it appears in the positive class.
- To get the negative frequency of any word in the vocabulary, we have to count how often it appears in the negative class.

Given the corpus and the labels above, we can build the following table:

| $V$     | $n_{pos}$ | $n_{neg}$ |
| ------- | --------- | --------- |
| I       | 3         | 3         |
| am      | 2         | 2         |
| happy   | 2         | 0         |
| sad     | 0         | 2         |
| because | 1         | 1         |
| love    | 1         | 0         |
| hate    | 0         | 1         |
| the     | 1         | 1         |
| weather | 1         | 1         |

We can observe that some words take clear sides, like `happy` and `sad`, while others are neutral, like `am` and `the`.

!!! example

    - The word `happy` appears twice in the positive class and zero times in the negative class.
    - The word `sad` appears zero times in the positive class and twice in the negative class.
    - The word `because` appears once in the positive class and once in the negative class.
    - The word `I` appears three times in the positive class and three times in the negative class.

!!! info

    In practice, there are multiple ways to implement such a table.

    - Pandas DataFrame
    - Python dictionary
    - Numpy array
    - ...

Based on this table, we can build the feature vector for a document $i$, by summing up the positive and negative frequencies of each word in the document.

Considering the **bias unit** as the first feature, the feature vector $x_i$ looks like this:

<!-- prettier-ignore-start -->
$$ x_i = [1, \sum_{j=1}^{m} n_{pos}(w_j), \sum_{j=1}^{m} n\_{neg}(w_j)] $$
<!-- prettier-ignore-end -->

<!-- TODO EXAM -->
!!! example

    Given the table of frequencies above, let's consider the following document:

    $$ \text{I am happy because the sun is shining} $$

    To get the feature vector, we sum up the frequencies of every word in the document per class:

    - $n_{pos} = 3 + 2 + 2 + 1 + 1 = 9$
    - $n_{neg} = 3 + 2 + 0 + 1 + 1 = 7$

    Considering the **bias unit** as the first feature, the feature vector $x_i$ for this document is:

    $$ x_i = [1, 9, 7] $$

!!! question

    How would you treat the words that are **not** in the vocabulary?
