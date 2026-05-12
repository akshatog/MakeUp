import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from src import config

def build_model(vocab_size: int, max_seq_len: int) -> Sequential:
    """Build and compile the Next Word Prediction LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM, input_length=max_seq_len-1),
        LSTM(config.LSTM_UNITS_1, return_sequences=True),
        Dropout(config.DROPOUT_RATE),
        LSTM(config.LSTM_UNITS_2),
        Dense(vocab_size, activation='softmax')
    ], name='NextWordLSTM')
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        metrics=['accuracy']
    )
    return model
