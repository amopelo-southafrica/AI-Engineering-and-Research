# Business Slogan Generation and Classification with LSTMs

## What this project is about

**Question:** Can a model generate a business slogan for a chosen industry, and can a second model identify the industry from that slogan?

I developed a two-part natural-language-processing experiment. One model generates slogan text word by word, conditioned on an industry category; the other predicts the industry associated with a slogan. The project explores how language models and classifiers can be combined in a simple business-text workflow.

## What I implemented

- Prepared slogan text using spaCy, including lowercasing, token cleaning, start/end markers and integer token sequences.
- Split the dataset into training and test sets before creating next-word training examples.
- Built LSTM-based models in TensorFlow/Keras for slogan generation and industry classification.
- Added functions to generate text, classify slogans and compare the requested industry with the predicted industry.
- Added training and model-evaluation steps for both models and a comparison of generated slogans against predicted industries.

## Technology and methods

Python, pandas, NumPy, spaCy, TensorFlow/Keras and scikit-learn; NLP preprocessing, sequence modelling, LSTMs and text classification.

## How to run

Install the libraries in `requirements.txt` and a compatible Jupyter environment. Download spaCy's `en_core_web_sm` language model if required. Place the appropriately licensed slogan dataset in the notebook environment, update `DATA_PATH`, and run `neural_network_nlp_capstone.ipynb`.

## What this demonstrates

Understanding of text preprocessing, sequential neural networks, training/evaluation separation and combining two models into one workflow.

**Data:** The notebook expects the slogan dataset to be provided through its `DATA_PATH` setting.
