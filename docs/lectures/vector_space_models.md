# Vector Space Models

Vector space models are a way of representing the meaning of words in a document. They are a fundamental concept in NLP, and are used in many applications such as document classification, information retrieval, and question answering.

Using vector space models, we can capture similarities, differences, dependencies or many other relationships between words.

!!! example

    The following sentence have very **similar words**, but they have **different meanings**:

    > Where are you **from**?

    > Where are you **going**?

    In contrast, the following sentences have very **different words**, but they have **similar meanings**:

    > What is your age?

    > How old are you?

    Vector space models can be used to capture such similarities and differences between words.

!!! example

    In the following sentence, the word **cereal** and the word **spoon** are related.

    > You eat **cereal** with a **spoon**.

    In the following sentence, the word **sell** depends on the word **buy**.

    > You **sell** something to someone who **buys** it.

Vector space models help us to capture such and many other relationships between words.

!!! quote "John Firth, 1957"

    **You shall know a word by the company it keeps.**

    This is one of the most fundamental concepts in NLP. When using vector space models, the way that representations are made is by identifying the context around each word in the text, which captures the relative meaning.

When learning these vectors, we usually make use of the neighboring words to extract meaning and information about the center word.

If we would cluster vectors together, we can observe that adjectives, nouns, verbs, etc. tend to be near to one another.

!!! question

    In vector space models, synonyms and antonyms are very close to one another. Why do you think this is the case?

<!--
Answer: this is because you can easily interchange them in a sentence, and they tend to have similar neighboring words!

Synonyms:
I bought a new automobile last week.
I bought a new car last week.

Antonyms:
She considered him her enemy
She considered him her friend
-->

## Co-Occurrence Matrix

From the co-occurrence matrix, we can extract the word vectors.

The vector representation of a word is called a **word embedding**.

We can use those word embeddings to find relationships between words.

In the following, we will look at two different approaches to create word embeddings.

![Vector space models workflow](../img/vector-space-models-workflow.drawio.svg)

!!! info

    The terms **word vector** is often used interchangeably with **word embedding**.
    Both terms refer to a numerical representation of words in a continuous vector space.

## Word by Word Design

In the word by word design, the **co-occurrence matrix** counts the number of times that a word appears in the context of other words within a given window size $k$.

!!! example

    Suppose we have the following two sentences:

    > I like simple data

    > I prefer simple and raw data

    With a window size of $k=2$, the co-occurrence matrix would look as follows:

    |   | I | like | prefer | simple | and | raw | data |
    | - | - | ---- | ------ | ------ | --- | --- | ---- |
    | I | 0 | 1    | 1      | 2      | 1   | 1   | 1    |
    | like | 1 | 0    | 0      | 1      | 0   | 0   | 1    |
    | prefer | 1 | 0    | 0      | 1      | 0   | 0   | 1    |
    | simple | 2 | 1    | 1      | 0      | 1   | 1   | 2    |
    | and | 1 | 0    | 0      | 1      | 0   | 0   | 1    |
    | raw | 1 | 0    | 0      | 1      | 0   | 0   | 1    |
    | data | 1 | 1    | 1      | 2      | 1   | 1   | 0    |

    If we look more closely at the word **data**, we can see that it appears in the context of the word **simple** twice, and in the context of the word **and** once, given a window size of $k=2$.

    So the word **data** can be represented as the following vector:

    $$x_{data} = [1, 1, 1, 2, 1, 1, 0]$$

    Note that the vector is of size $n$, where $n$ is the number of unique words in the vocabulary.

## Word by Document Design

For a word by document design, the process is quite similar.

But instead of counting co-occurrences of words, we count the number of times that a word appears in documents of a specific category.

!!! example

    Let's assume our corpus contains documents of three categories:

    - entertainment
    - economy
    - machine learning

    For the words **data** and **movie**, we could assume the following counts per category:

    |   | entertainment | economy | machine learning |
    | - | ------------- | ------- | ---------------- |
    | data | 1000 | 4000 | 9000 |
    | movie | 7500 | 2000 | 500 |

    So the word **data** can be represented as the following vector:

    $$x_{data} = [1000, 4000, 9000]$$

    And the word **movie** can be represented as the following vector:

    $$x_{movie} = [7500, 2000, 500]$$

    Note that the vector is of size $n$, where $n$ is the number of categories.

We could visualize those vectors in a vector space as follows:

![Word by document design](../img/vector-space-models-word-by-document.drawio.svg)

!!! note

    For the sake of drawing, instead of the word vectors, the figure shows the vectors of the categories.

    However, we should see similar results if we would draw the word vectors in a three-dimensional vector space.

## Eucledian Distance

The **Euclidean distance** between two vectors $\mathbf{x}$ and $\mathbf{y}$ in an n-dimensional space is defined as follows:

$$
\begin{align}
d &= \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots + (x_n - y_n)^2} \\
&= \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
\end{align}
$$

It is the **straight line** between two points in an n-dimensional space, and derived from the Pythagorean theorem.

We can also interpret it as the **length of the vector** that connects the two points (aka the **norm** of the vector).

!!! quote "Pythagoraen Theorem"

    In a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.

    $$
    c^2 = a^2 + b^2
    $$

Using the Euclidean distance, we can calculate **how close** two vectors are and thus, use it as a **similarity metric**.

![Euclidean distance](../img/vector-space-models-euclidean-distance.drawio.svg)

!!! example

    If we continue with the example from above, we can calculate the Euclidean distance between the **machine learning** category vector $\mathbf{ml}$ and the **entertainment** category vector $\mathbf{e}$ as follows:

    Let the two vectors be:

    $$
    \begin{align}
    \mathbf{ml} &= [9000, 500] \\
    \mathbf{e} &= [1000, 7500]
    \end{align}
    $$

    Then the Euclidean distance is calculated as follows:

    $$
    \begin{align}
    d &= \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2} \\
    &= \sqrt{(ml_1 - e_1)^2 + (ml_2 - e_2)^2} \\
    &= \sqrt{(9000 - 1000)^2 + (500 - 7500)^2} \\
    &= 8000
    \end{align}
    $$

As you can see from the formula, this generalizes to **any number of dimensions**.

!!! example

    Let's assume we have the following **co-oocurrence matrix**, and we want to calculate the Euclidean distance between the words **beer** and **pizza**.

    |   | data | beer | pizza |
    | - | ---- | ---- | ------ |
    | AI | 6 | 0    | 1      |
    | drinks | 0 | 4    | 6      |
    | food | 0 | 6    | 8      |

    Based on this co-occurrence matrix, we can represent the words **beer** and **pizza** as the following vectors:

    $$
    \begin{align}
    \mathbf{b} &= [0, 4, 6] \\
    \mathbf{p} &= [1, 6, 8]
    \end{align}
    $$

    Then the Euclidean distance is calculated as follows:

    $$
    \begin{align}
    d &= \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2} \\
    &= \sqrt{(b_1 - p_1)^2 + (b_2 - p_2)^2 + (b_3 - p_3)^2} \\
    &= \sqrt{(0 - 1)^2 + (4 - 6)^2 + (6 - 8)^2} \\
    &= \sqrt{1 + 4 + 4} \\
    &= 3
    \end{align}
    $$

In Python, we can calculate the Euclidean distance using the `numpy.linalg.norm` function.

```python
import numpy as np

# Define two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Calculate the Euclidean distance
distance = np.linalg.norm(a - b)

print("Euclidean distance:", distance)
```

!!! note

    The [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) function calculates the norm of a vector or matrix.

    Since the **norm of a vector is its length**, the function can be used to calculate the Euclidean distance between two vectors.

## Cosine Similarity
