import os
from pathlib import Path

# Project Roots
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data paths
RAW_DATA_PATH = DATA_DIR / "dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_dataset.pkl"

# Model paths
MODEL_SAVE_PATH = MODELS_DIR / "next_word_model.keras"
TOKENIZER_SAVE_PATH = MODELS_DIR / "tokenizer.pkl"

# Hyperparameters
MAX_SEQUENCE_LENGTH = None  # Will be set dynamically during data prep
EMBEDDING_DIM = 100
LSTM_UNITS_1 = 150
LSTM_UNITS_2 = 100
DROPOUT_RATE = 0.2

# Training params
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 100
