import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src import config
from src.data import load_data, prepare_data, save_tokenizer
from src.model import build_model

def train():
    print("Loading data...")
    df = load_data(config.RAW_DATA_PATH)
    corpus = df['sentence'].tolist()
    
    print("Preparing data...")
    X, y_cat, tokenizer, vocab_size, max_seq_len = prepare_data(corpus)
    
    print("Saving tokenizer...")
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    save_tokenizer(tokenizer, config.TOKENIZER_SAVE_PATH)
    
    print(f"Building model (Vocab Size: {vocab_size}, Max Seq Len: {max_seq_len})...")
    model = build_model(vocab_size, max_seq_len)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True, min_delta=0.005),
        ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, min_lr=1e-5, verbose=1)
    ]
    
    print("\n🚀 Starting training...")
    history = model.fit(
        X, y_cat,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    
    print(f"\n💾 Saving model to {config.MODEL_SAVE_PATH}")
    model.save(config.MODEL_SAVE_PATH)
    
    print("Training complete!")

if __name__ == "__main__":
    train()
