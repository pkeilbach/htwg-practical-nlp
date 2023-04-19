# Syllabus

## Part 1 - NLP Basics

Lectures: 1

The first part introduces NLP, with all important terminology for the course. It also covers some basic preprocessing aspects, as well as feature engineering.

### NLP Intro

- What this course is about
- learning goals of this course
- high level overview of NLP
- NLP terminology and use cases
- ...

### Preprocessing

- how to work with text data, cleaning
- tokenization
- stop words
- stemming
- ...

### Feature Engineering

- Common features to use in NLP
- TF/IDF
- Positive and negative word frequency
- ...

### Lab: Python Generators and Pre-Processing

- Warm-Up: Python generators and other programming concepts that are used throughout the lecture
- implement functions for preprocessing tasks, and to transform text into features

## Part 2 - Classification and Vector Space

Lectures: 1,5

This part will build the bridge from NLP basics to basic machine learning approaches. It will show how we can transform text into vectors and apply machine learning models on it. Students will learn or recap simple yet powerful machine learning concepts.

### Logistic Regression

- Overview / recap of logistics regression
- training / testing / cost function
- ...

### Vector Space Models

- Euclidean Distance
- Cosine similarity
- ...

### K-nearest neighbors

- explaining the algorithm
- optimizing K-nearest neighbors with locality sensitive hashing
- ...

### Lab: Sentiment Analysis

- Sentiment analysis with logistic regression
- Word embeddings and machine translation
- Document search with K-nearest neighbors

## Part 3 - Probabilistic NLP models

Lectures: 1,5

This part covers probabilistic NLP approaches, and familiarizes with the mathematical concepts behind them. This part should get a little more weight, as the statistical methods are interesting, still somewhat intuitive, and at the same time rather easy to implement. It will show the students that even with leight-weight approaches, we can solve impressing NLP tasks.

### Naive Bayes

- Bayes Rule
- Log Likelihood
- Laplacian Smoothing
- ...

### Minimum Edit Distance

- explain the minimum edit distance algorithm for auto correct
- ...

### Optional: Hidden Markov Models

- Markov Chains
- Hidden Markov Models
- Viterbi Algorithm
- ...

### N-Grams and Sequence Probabilities

- explaining N-grams and sequence probabilities
- Continuous Bag of Words model
- activation functions: Softmax and ReLU
- ...

### Lab: Auto-correction and Auto-completion

- Autocorrect with minimum edit distance
- Markov Chains for Part of Speech Tagging
- Autocomplete with N-Grams

## Part 4 - Advanced NLP Concepts

Lectures: 3

This part covers somewhat advanced topics, they should be only explained on a high level. It is more about transporting the idea of how SOTA NLP models work, without going much into architectural detail. Additionally, practical aspects should be covered, to get a feeling for real-life NLP scenarios.

### Sequence Models

- Vanishing Gradient Problem
- Recurrent Neural Networks
- Long Short-Term memory
- ...

### Attention Models

- From RNN to Transformers
- Attention concept
- Attention models: GPT-3, BERT, T5
- GLUE and BLUE metrics
- ...

### Production Scale NLP

- How to work with large amounts of text data in practice,
- how to efficiently store and retrieve text data
- A selection of technologies should be presented, i.e. common NLP libraries and open source projects, or document oriented database technologies like ElasticSearch
- ...

### Lab: NLP Use Case

- Advanced NLP use case (e.g. text summarization, Q&A, or chatbot) with an advanced model like BERT or GPT-3
- Document search and NLP with ElasticSearch

## Literature

Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit. O'Reilly Media. <https://www.oreilly.com/library/view/natural-language-processing/9780596803346/> Also available open source: <https://www.nltk.org/book_1ed/>

Jurafsky, D., & Martin, J. H. (2008). Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition (2nd ed.). Prentice Hall. <https://www.amazon.de/dp/0131873210> Draft of 3rd edition also available open source: <https://web.stanford.edu/\~jurafsky/slp3/>

Tunstall, L., Von Werra, L., & Wolf, T. (2022). Natural Language Processing with Transformers. O'Reilly Media. <https://www.oreilly.com/library/view/natural-language-processing/9781098136789/>

Vajjala, S., Majumder, B., Gupta, A., & Surana, H. (2020). Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems. O'Reilly Media. <https://www.oreilly.com/library/view/practical-natural-language/9781492054047/>
