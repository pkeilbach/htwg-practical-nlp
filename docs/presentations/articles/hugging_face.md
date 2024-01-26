# Hugging Face

_Author: [Luis Nothvogel](mailto:luis.nothvogel@htwg-konstanz.de)_  
  
## TL;DR  
  
Hugging Face has emerged as a pivotal player in the AI and machine learning arena, specializing in natural language processing (NLP). This article delves into its core offerings, including model hosting, spaces, datasets, pricing, and the Terraformer API. Hugging Face is not only a repository for cutting-edge models but also a platform for collaboration and innovation in AI.  

## Model Hosting on Hugging Face

Hugging Face has made a name for itself in model hosting. It offers a vast repository of pre-trained models, primarily focused on NLP tasks such as text classification, question answering, and language generation. According to them they host over 350k models. Users can easily download and deploy these models. Moreover, Hugging Face allows users to upload and share their models with the community.

```python
from transformers import pipeline, set_seed

# Example of using a pre-trained model
generator = pipeline('text-generation', model='gpt2')  
set_seed(42)  
generated_texts = generator("The student worked on", max_length=30, num_return_sequences=2)  
print(generated_texts)
```

This outputs the following:
```python
[{'generated_text': 'The student worked on his paper, which you can read about here. You can get an ebook with that part, or an audiobook with some of'}, {'generated_text': 'The student worked on this particular task by making the same basic task in his head again and again, without the help of some external helper, even when'}]
```

## Spaces: A Collaborative Environment

Spaces are an innovative feature of Hugging Face, offering a collaborative environment where developers can build, showcase, and share machine learning applications. Users can deploy models as web applications, creating interactive demos that are accessible to a broader audience. Spaces support various frameworks like Streamlit and Gradio. According to them they host over 150k spaces.

## Diverse Datasets at Your Disposal

The Hugging Face ecosystem includes a wide range of datasets, catering to different NLP tasks. The Datasets library simplifies the process of loading and processing data, ensuring efficiency and consistency in model training. According to them they host over 75k datasets.

[Wikipdia Referenz](https://huggingface.co/datasets/wikimedia/wikipedia)
```python
from datasets import load_dataset

# Example of loading a dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.en")
```


## Transformers API: Transform Text Effortlessly

The Transformers API is a testament to Hugging Face's innovation. This API simplifies the process of text transformation, making it accessible even to those with limited programming skills. It supports a variety of NLP tasks and can be integrated into various applications.

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

## Tokenizers

Hugging face tokenisers are an essential component of modern natural language processing. These tokenisers are characterised by their speed and flexibility and support a variety of tokenisation methods such as BPE (Byte-Pair-Encoding), WordPiece or SentencePiece. One of the key features of Hugging Face Tokenisers is their compatibility with different language models such as BERT, GPT and RoBERTa, making them a universal tool in the NLP community. With ongoing innovations and updates, Hugging Face Tokenisers remain at the forefront of NLP technology by providing continuously improved efficiency and accuracy in language processing.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```

## Inference

Hugging Face Inference plays a crucial role in turning trained language models into productive applications. The platform provides an intuitive and powerful infrastructure for inferencing models, which means that developers can easily access pre-trained models to generate real-time predictions for a wide range of NLP tasks. Thanks to its efficient implementation and support for hardware acceleration technologies, Hugging Face Inference enables the seamless integration of language models into applications ranging from chatbots to machine translation and sentiment analysis.

The Inference API Url is always defined like this: 
```python
ENDPOINT = https://api-inference.huggingface.co/models/<MODEL_ID>
```

Example in Python with gpt2:
```python
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query("Write a story about a Knight")
```

## Key Takeaways

- Hugging Face is at the forefront of NLP, offering a wealth of models, datasets, and tools.
- Its **model hosting** platform is robust, user-friendly, and widely adopted in the AI community.
- **Spaces** foster **collaboration and accessibility**, allowing users to easily share and demonstrate their ML applications.
- The platform's commitment to providing diverse **datasets** accelerates research and development in NLP.
- The **Terraformer API** is a notable tool for simplifying text transformations, enhancing the accessibility of NLP.

## References

- [Hugging Face: The AI community building the future.](https://HuggingFace.co/)
- [Hugging Face Transformers Documentation](https://HuggingFace.co/docs/transformers/index)
- [Hugging Face Datasets Library](https://HuggingFace.co/docs/datasets/index)
- [Hugging Face Inference](https://huggingface.co/docs/api-inference/index)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)
