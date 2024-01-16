# Sequence Models

## Limitations of N-Gram Language Models

Recall that N-gram language models are used to compute the probability of a sequence of words. For that, we need to compute the conditional probability of a word given the $N-1$ previous words. This approach has two main limitations:

- N-gram models consider only a fixed number of preceding words (n-1 context) to predict the next word, and thus, have **limited contextual information**. This limitation results in the model being unable to capture **long-range dependencies** or understand the context beyond the immediate history.
- To capture dependencies of words that are very distant from each other, we need to use a large $N$. This can be difficult to estimate without a large corpus. In practice, this can lead to **sparsity** issues, where many possible n-grams may not be observed in the training data.
- Even with a large corpus, such a model would require a lot of **memory** to store the counts of all possible $N$-grams.

So for large $N$, this becomes very impractical. A type of model that can help us with this is the Recurrent Neural Network (RNN).

!!! example "long-range dependencies"

    Consider the following sentence:

    > Mary was supposed to study with me. I called her, but she did not <?>.

    Where the expected word is "answer".

    A traditional language model (let's say trigram) would probably predict a word like "have", since we can assume that the combination "did not have" is very frequent (or at least more frequent than the word "answer") in regular corpora.

    We would need a very large $N$ to capture the dependency between "did not" and "answer", as we would also need to consider the beginning of the sentence "I called her".

!!! example "Penta-gram"

    To predict the probability of a five-word sequence using a penta-gram model, we can use the following formula:

    $$P(w_1, w_2, w_3, w_4, w_5) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) \cdot P(w_4 | w_1, w_2, w_3) \cdot P(w_5 | w_1, w_2, w_3, w_4)$$

    We can easily imagine that the larger the $N$, the more sparse the data will be, and the more difficult it will be to estimate the probabilities for the N-grams.

    Note that this is the [sequence probability](./language_models.md#sequence-probabilities) (which makes use of the [Markov assumption](./language_models.md#markov-assumption)), and not the [N-gram probability](./language_models.md#n-gram-probability).

## Recurrent Neural Networks

RNNs propagate information from the beginning of a sequence through to the end. This allows them to capture long-range dependencies.

We can see this process illustrated in the following figure:

![Illustration of a Recurrent Neural Network](../img/sequence-models-rnn.png)

- Each of the boxes represent the values computed at each particular step.
- The colors represent the information that is propagated through the network.
- The arrows indicate how the information is propagated.
- The information from every word in the sequence is multiplied by the input weight matrix $W_x$.
- To propagate information from one step to the next, we multiply the information from the previous step by the hidden state weight matrix $W_h$.
- The hidden state at each time step is computed as a function of the previous hidden state $h_{t-1}$ and the input at the current step $x_t$.

The hidden states are what allow the RNN to capture long-range dependencies. As we can see, in the last step, there is still information from the first step. This is what allows the RNN to capture long-range dependencies.

!!! info

    The weights $W_x$ and $W_h$ are shared across all steps, that means we only need to learn them once, and then we can apply them to every step.

    This is why the RNN is called a **recurrent** neural network, because it performs the same task for every element of a sequence, with the output being dependent on the previous computations.

!!! info "Loss Function"

    For RNNs, typically the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss function is used.

!!! example "Implementation Note"

    Tensorflows [`tf.scan`](https://www.tensorflow.org/api_docs/python/tf/scanz) function can be used to implement RNNs. It takes a function and applies it to all elements of a sequence. You can also pass an optional initializer, which is used to initialize the first element of the sequence.

    A variant is given here:

    ```python
    def scan(fn, elems, weights, h_0=None):
        h_t = h_0
        ys = []
        for e in elems:
            y, h_t = fn([e, h_t], weights)
            ys.append(y)
        return ys, h_t
    ```

    Note that in Python, you can pass a function as an argument to another function.

!!! note "Gated Recurrent Units"

    The [Gated Recurrent Unit (GRU)](https://en.wikipedia.org/wiki/Gated_recurrent_unit) is a variant of the RNN that is easier to train, and often performs better in practice.

    It is similar to the RNN, but it has two gates:

    - The **update gate** controls how much of the previous state is kept.
    - The **reset gate** controls how much of the previous state is forgotten.

## Bi-directional RNNs

In a bi-directional RNN, information flows in both directions.

- The forward RNN propagates information from the beginning of the sequence to the end.
- The backward RNN propagates information from the end of the sequence to the beginning.

With this, a bi-directional RNN can capture information from both the past and the future context.

Note that the computations of the forward and backward RNNs are independent of each other, and thus, can be parallelized.

!!! example

    Given the sentence:

    > I was trying really hard to get a hold of <?>. Louise finally answered when I was about to give up.

    Since Louise doesn't appear until the beginning of the second sentence, a regular RNN would have to guess between "her", "him", or "them". However, a bi-directional RNN would be able to capture the context from the end of the sequence, and thus, would be able to predict "her" with a higher probability.

!!! info "Deep RNNs"

    Another variant of RNNs are deep RNNs.

    Similar to deep neural networks, d eep RNNs are RNNs with multiple hidden layers. They can be used to capture more complex patterns. We can think of them as multiple RNNs stacked on top of each other.

    1. Get the hidden state for the current layer (propagate information through time). ➡️
    2. Use the hidden state of the current layer as input for the next layer (propagate information through layers). ⬆️

## Vanishing Gradient Problem

## LSTMs
