import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, Dict, Any
from src import config

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from csv."""
    return pd.read_csv(filepath)

def clean_corpus(corpus: list) -> list:
    """Lowercase and remove punctuation from the corpus."""
    return [re.sub(r"[^a-z\s]", "", str(s).lower().strip()) for s in corpus]

def prepare_data(corpus: list) -> Tuple[np.ndarray, np.ndarray, Tokenizer, int, int]:
    """
    Prepare sequences for training.
    Returns: X, y_cat, tokenizer, vocab_size, max_seq_len
    """
    corpus_clean = clean_corpus(corpus)
    
    # Tokenization
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(corpus_clean)
    vocab_size = len(tokenizer.word_index) + 1
    
    # N-gram Sequence Generation
    input_sequences = []
    for line in corpus_clean:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i+1])
            
    # Padding
    max_seq_len = max(len(s) for s in input_sequences)
    padded = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))
    
    # X and y
    X = padded[:, :-1]
    y = padded[:, -1]
    y_cat = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
    
    return X, y_cat, tokenizer, vocab_size, max_seq_len

def save_tokenizer(tokenizer: Tokenizer, filepath: str) -> None:
    """Save tokenizer to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(filepath: str) -> Tokenizer:
    """Load tokenizer from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    df = load_data(config.RAW_DATA_PATH)
    corpus = df['sentence'].tolist()
    X, y, tokenizer, vocab_size, max_seq_len = prepare_data(corpus)
    print(f"Vocab Size: {vocab_size}, Max Seq Len: {max_seq_len}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
