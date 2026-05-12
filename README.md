# Next Word Prediction with LSTM 🧠

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A professional, production-ready implementation of a **Next Word Prediction Model** using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) neural networks. 

This repository provides an end-to-end workflow, from raw data processing to model training and interactive prediction inference.

---

## 📂 Project Structure

```text
MakeUp/
├── data/                       # Datasets
│   ├── raw/                    # Raw source data
│   ├── processed/              # Processed datasets and tokenized sequences
│   ├── dataset.csv             # The main CSV dataset
│   └── dataset.txt             # The text format dataset
├── models/                     # Saved Keras models and Tokenizers
├── notebooks/                  # Jupyter notebooks for exploration and prototyping
│   └── next_word_prediction_final.ipynb
├── src/                        # Source code for the project
│   ├── __init__.py             # Makes src a Python module
│   ├── config.py               # Hyperparameters and path configurations
│   ├── data.py                 # Data loading, cleaning, and preprocessing
│   ├── model.py                # LSTM neural network architecture definition
│   ├── train.py                # Script to execute model training
│   └── predict.py              # Prediction and autocomplete utilities
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup file for package installation
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### 1. Clone & Setup Environment

It's recommended to use a virtual environment:

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

You can install the required packages and the local project as a package:

```powershell
# Install the required packages
pip install -r requirements.txt

# Install the source code as a local editable package
pip install -e .
```

### 3. Train the Model

To train the model from scratch using the dataset in `data/dataset.csv`:

```powershell
python -m src.train
```

This will:
- Parse and preprocess the text.
- Build the token vocabulary.
- Train the LSTM model.
- Save the trained model to `models/next_word_model.keras`.
- Save the tokenizer to `models/tokenizer.pkl`.

### 4. Run Predictions

Once the model is trained, you can predict the next word using the prediction CLI:

```powershell
# Predict the next 3 probable words
python -m src.predict "artificial intelligence is" --top_n 3

# Auto-complete a sentence by 5 words
python -m src.predict "machine learning can" --complete 5
```

## 🧠 Model Architecture

The model uses a multi-layer architecture tailored for sequence prediction:
- **Embedding Layer**: Converts the tokenized words into dense vectors.
- **LSTM Layer 1**: Extracts complex patterns across the input sequence.
- **Dropout Layer**: Prevents overfitting.
- **LSTM Layer 2**: Further refines sequence prediction.
- **Dense Layer**: Outputs a probability distribution over the vocabulary using `softmax`.

## 📊 Notebooks

If you prefer an interactive environment with visualizations (training graphs, word clouds), check out the `notebooks/next_word_prediction_final.ipynb`.

## 🛠️ Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is open-source and available under the MIT License.
