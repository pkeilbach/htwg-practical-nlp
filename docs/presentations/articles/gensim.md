# Gensim

_Author: [Fabian Renz](mailto:fa721ren@htwg-konstanz.de)_

## TL;DR

Gensim is a powerful Python library for natural language processing, offering scalable implementations of various algorithms for topic modeling, document similarity analysis, and word embeddings.

## Introduction

Natural Language Processing (NLP) has become an integral part of various applications, from chatbots to sentiment analysis. Gensim, an open-source Python library, provides tools for processing and analyzing large text corpora efficiently.

## Understanding Gensim

### Topic Modeling with Latent Dirichlet Allocation (LDA)

Gensim simplifies the implementation of Latent Dirichlet Allocation, a popular algorithm for topic modeling. We'll walk through the process of creating a bag-of-words representation, building an LDA model, and interpreting the discovered topics.

```python
# Sample documents
documents = [
    "Gensim is a powerful Python library for natural language processing.",
    "It provides tools for topic modeling, document similarity analysis, and word embeddings.",
    "In this article, we will explore key features of Gensim and demonstrate its usage.",
]

# Tokenize the documents
tokenized_documents = [doc.split() for doc in documents]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Build the LDA model
# creating num_topics=2 topics
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print the topics
for topic in lda_model.print_topics():
    print(topic)
```

Output (2 Topics):

```python
(0, '0.059*"and" + 0.035*"key" + 0.035*"we" + 0.035*"usage." + 0.035*"its" + 0.035*"of" + 0.035*"this" + 0.035*"features" + 0.035*"will" + 0.035*"explore"')
(1, '0.057*"for" + 0.057*"Gensim" + 0.056*"a" + 0.056*"is" + 0.056*"language" + 0.056*"natural" + 0.056*"library" + 0.056*"processing." + 0.056*"Python" + 0.056*"powerful"')
```

The weight of each word represents the importance of the word in each topic. Given the limited data we used, the weights are very similar.

### Word Embeddings with Word2Vec

Word embeddings are crucial for capturing semantic relationships between words. Gensim's Word2Vec implementation allows us to learn these embeddings from large text corpora. We'll demonstrate how to train a Word2Vec model and utilize the resulting word vectors.

```python
from gensim.models import Word2Vec

# Training a Word2Vec model
# creating vector with vector_size=10
word2vec_model = Word2Vec(tokenized_documents, vector_size=10, window=5, min_count=1, workers=4)

# Accessing the vector for a specific word
vector_for_word = word2vec_model.wv['Gensim']
```

Output (Vector with 10 dimensions for the word 'gensim'):

```python
[-0.00538505  0.0023335   0.05105938  0.09008791 -0.09300902 -0.07120689
  0.06456329  0.08967262 -0.05014846 -0.03767093]
```

Each floating-point value corresponds to a dimension in the embedding space.

### Document Similarity Analysis

Gensim facilitates the calculation of document similarity using techniques like cosine similarity. We'll explore how to measure similarity between documents and discuss its applications in information retrieval and document clustering.

```python
from gensim import similarities

# Transform corpus to TF-IDF space
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Create a similarity index
index = similarities.MatrixSimilarity(corpus_tfidf)

# Calculate similarity between documents
sims = index[corpus_tfidf]

# Print the document similarity matrix
print(list(enumerate(sims)))
```

Output (similarity between documents):

```python
[document_number, similarity to [document_0, document_1, document_2]]
[(0, array([0.99999994, 0.01477628, 0.01351875], dtype=float32)),
(1, array([0.01477628, 0.9999999 , 0.01213155], dtype=float32)),
(2, array([0.01351875, 0.01213155, 0.99999994], dtype=float32))]
```

## Key Takeaways

- Gensim provides scalable and efficient tools for NLP tasks.
- LDA is valuable for uncovering latent topics in a collection of documents.
- Word2Vec enables the creation of meaningful word embeddings.
- Document similarity analysis aids in tasks such as information retrieval.

## References

- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Gensim: LDA](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)
- [Gensim: Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Gensim: Similarity](https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html)
