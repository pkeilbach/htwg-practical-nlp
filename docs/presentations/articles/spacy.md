# SpaCy

Author: Lukas Brandt

## TL;DR

SpaCy offers NLP processing capabilities through pre-trained Pipelines. These Pipelines process the Text and extract information from it, which can be used for further processing or training of LLMs or Chatbots. The pre-trained pipelines can also be customized and trained with additional data.

This article aims to provide a brief overview of spaCy without delving into excessive detail. Relevant links for further information are included.

## Introduction

SpaCy is a Python package developed by ExplosionAI GmbH. SpaCy is considered 'production ready' and is open-source.

It provides pre-trained pipelines for many languages, including components for dependency parsing, sentence segmentation, lemmatization, named entity recognition, tokenization, rule-based matching, and more. When provided with text, it returns a 'doc' object. This object contains all the information extracted from the text by the pipeline. This information can be used e.g. in machine learning, LLMs or chatbots.

## Pipelines

The pre-trained pipelines follow a naming convention: **lang**\_**type**\_**genre**\_**size**.

| Part  | Values          | Description                  |
| :---- | :-------------- | :--------------------------- |
| lang  | en, de, fi, ... | Language                     |
| type  | core, dep, ...  | Capabilities of the pipeline |
| genre | news, web, ...  | type of training text        |
| size  | sm, md, lg, trf | Size                         |

The _transformer_ sized pipelines are the largest but also the most accurate pipelines.

### Pre-trained pipelines

What these pretrained pipelines look like in more detail:

![pre-trained pipeline](https://spacy.io/images/pipeline.svg)

| Name       | Component         | Creates                                           | Description                                      |
| :--------- | :---------------- | :------------------------------------------------ | :----------------------------------------------- |
| tokenizer  | Tokenizer         | Doc                                               | Segment text into tokens.                        |
| tagger     | Tagger            | Token.tag                                         | Assign part-of-speech tags.                      |
| parser     | DependencyParser  | Token.head, Token.dep, Doc.sents, Doc.noun_chunks | Assign dependency labels.                        |
| ner        | EntityRecognizer  | Doc.ents, Token.ent_iob, Token.ent_type           | Detect and label named entities.                 |
| lemmatizer | Lemmatizer        | Token.lemma                                       | Assign base forms.                               |
| textcat    | TextCategorizer   | Doc.cats                                          | Assign document labels.                          |
| custom     | custom components | Doc._.xxx, Token._.xxx, Span.\_.xxx               | Assign custom attributes, methods or properties. |

For further reading on Doc objects and pipelines check [here](https://spacy.io/usage/spacy-101#pipelines).

### Training and customization

The existing pipelines can be customized or further trained. To customize a pipeline certain parts of it can, for example, be deactivated:

```python
nlp_ner = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger",
                                                "parser", "attribute_ruler", "lemmatizer"])
```

This example deactivates every component in the pipeline, that is not needed for Named Entity Recognition.

If further training is desired. It is necessary to convert the training and validation data to spaCy's binary format: `.spacy`. To begin training a config needs to be created. This can either be done on the website itself oder via the `spacy init config config.cfg` command. This config contains everything needed, apart from the paths to the training and validation data.

Training is then commenced either via python:

```python
from spacy.cli.train import train

train("./config.cfg", overrides={"paths.train": "./train.spacy", "paths.dev": "./dev.spacy"})

```

or in the shell:

```sh
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy
```

To use this now trained pipeline, the command to load it changes to `nlp = assemble('config.cfg')`.

For further reading on training look [here](https://spacy.io/usage/spacy-101#training).

## Large-Language Models and spaCy

SpaCy can be used in conjunction with LLMs by utilizing the `spacy-llm` python package.This package supports self-hosted models built using pyTorch or TensorFlow, OpenAI's API, and models from Hugging Face.

The configuration file defines how to use these models. Here is an example configuration using OpenAI's GPT3.5 model.

```conf
[nlp]
lang = "en"
pipeline = ["llm"]

[components]

[components.llm]
factory = "llm"

[components.llm.task]
@llm_tasks = "spacy.TextCat.v2"
labels = ["COMPLIMENT", "INSULT"]

[components.llm.model]
@llm_models = "spacy.GPT-3-5.v1"
config = {"temperature": 0.0}
```

For further reading on LLM's and spacy look [here](https://spacy.io/usage/large-language-models).

## Example

```sh
# Installation
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -U spacy

# Download of a pre trained pipeline
python -m spacy download en_core_web_sm
```

```py
import spacy

# Creating the NLP object
nlp = spacy.load("en_core_web_sm")

# Example text and using nlp
doc = nlp("This is an example sentence with numbers 234.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

# Output:
###
# This this PRON DT nsubj Xxxx True True
# is be AUX VBZ ROOT xx True True
# an an DET DT det xx True True
# example example ADJ JJ compound xxxx True False
# sentence sentence NOUN NN attr xxxx True False
# with with ADP IN prep xxxx True True
# numbers number NOUN NNS pobj xxxx True False
# 234 234 NUM CD nummod ddd False False
# . . PUNCT . punct . False False
###
```

## Key Takeaways

- spaCy is a python package used for NLP tasks.
- It offers many text processing capabilities that prepare the text for further usage.
- Pre-trained pipelines are availabile in many languages and sizes.
- Pre-trained pipelines can be customized and further trained.
- It is possible to use spaCy in conjunction with LLM's.
- The documentation is the best first place too look for answers as it is vast and contains multiple examples.
- Projects that use spaCy can be found [here](https://spacy.io/universe).

## References

- [spacy.io](https://spacy.io/)
- [GitHub](https://github.com/explosion/spaCy)
- [ExplisionAI](https://explosion.ai/)
