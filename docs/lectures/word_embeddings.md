# Word Embeddings

## Revisit One-Hot Encoding

In the lecture about feature extraction, we have seen that we can represent words as vectors using [one hot encoding](./feature_extraction.md#one-hot-encoding).

One-hot encoding is a very simple way to represent words as vectors, but it has some major **disadvantages**:

- the resulting vectors are very **high dimensional**, i.e. one dimension for each word in the vocabulary: $n_{\text{dim}} = |V|$

- it does not capture **meaning**, i.e. all words have the same distance to each other:

  ![One-Hot Encoding](../img/word-embeddings-one-hot-encoding-distance.png)

## Word Embeddings Overview

From the lecture about vector space models, we already know that similar words should be close to each other in the vector space.

But how can we achieve this? In this lecture we will learn about word embeddings, which are a way to represent words as vectors.

![Word Embeddings](../img/word-embeddings.drawio.svg)

In the figure above, we have two dimensional word embeddings:

- the first dimension represents the word's sentiment in terms of positive or negative.
- the second dimensions indicates whether the word is more concrete or abstract.

In the real world, word embeddings are usually much higher dimensional, e.g. 100 or 300 dimensions.

Each dimension represents a different aspect of the word. We do not know what exactly each dimension represents, but we know that similar words are close to each other in the vector space.

Word embeddings have the following advantages:

- they are dense, i.e. they do not contain many zeros
- they are low dimensional, i.e. they do not require much memory
- they allow us to encide meaning
- they capture semantic and syntactic information
- they are computationally efficient

!!! note

    Note that in the [word by word design](./vector_space_models.md#word-by-word-design), we have as many features as we have words in our vocabulary. This is not very efficient, because we have to store a lot of zeros. With word embeddings, we can reduce the number of features to a much smaller number, e.g. 100 or 300, while at the same time capturing the meaning of the words (which is not possible with the word by word design).

    We could also say we are giving up precision for gaining meaning.

!!! tip

    Here is a list of popular word embeddings methods/models:

    - [FastText (Facebook, 2016)](https://arxiv.org/abs/1607.04606)
    - [GloVe (Stanford, 2014)](https://nlp.stanford.edu/pubs/glove.pdf)
    - [Word2Vec (Google, 2013)](https://arxiv.org/abs/1301.3781)

    More sophisticated models use **advanced deep learning network architectures** to learn word embeddings.
    In these advanced models, words have different embeddings depending on their context (e.g. plant as flower or factory or as a verb).
    Here are some popular examples:

    - [ELMo (Allen Institute, 2018)](https://arxiv.org/abs/1802.05365)
    - [BERT (Google, 2018)](https://arxiv.org/abs/1810.04805)
    - [GPT-2 (OpenAI, 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

    There are also approaches to fine tune such models on your own corpus.

## Word Embeddings Process

Here is a high level overview of the word embeddings process:

![Word Embeddings Process](../img/word-embeddings-process.drawio.svg)

The [corpus](./language_models.md#text-corpus) are words in their context of interest, e.g. wikipedia, news articles, etc. This can be generic or domain specific.

!!! info "Corpus"

    Suppose we want to generate word embeddings based on Shakespeare's plays, then the corpus would be all of Shakespeare's plays, but not Wikipedia or news articles.

After **preprocessing**, we should have the words represented as **vectors**. Typically, we use one-hot encoding for this.

Those one-hot encoded vectors are then fed into the word **embeddings method**. This is usually a machine learning model, that performs a learning task on the corpus, for example, predicting the next word in a sentence, or predicting the center word in a context window.

The dimension of the word embeddings is one of the **hyperparameters** of the model which needs to be determined. In practice, it typically ranges from a few hundred to a few thousand dimensions.

!!! tip

    Remeber that it is the [context](./vector_space_models.md#introduction) that determines the meaning of a word

!!! info "Self-supervised Learning"

    The learning task for word embeddings is self-supervised, i.e. the input data is unlabeled, but the data itself provides the necessary context (which would otherwise be the labels), because we are looking at the context of a word.

    A corpus is a _self-contained_ data set, i.e. it contains all the information necessary to learn the embeddings, that is, the training data and the labels.

!!! info "Dimensions of Word Embeddings"

    Using a higher number of dimensions allows us to capture more nuances of the meaning of the words, but it also requires more memory and computational resources.

## Continuous Bag of Words (CBOW)

## Transforming Words to Vectors

## Architecure of CBOW Model

- Architecture Diagram showing input, hidden and output layers

## Evaluation of Word Embeddings

- Intrinsic vs. Extrinsic Evaluation
