"""Prediction functions for next word prediction."""
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple

from src.data import load_tokenizer
from src import config

class NextWordPredictor:
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """Initialize the predictor by loading the model and tokenizer."""
        model_path = model_path or config.MODEL_SAVE_PATH
        tokenizer_path = tokenizer_path or config.TOKENIZER_SAVE_PATH
        
        try:
            self.model = load_model(model_path)
            self.tokenizer = load_tokenizer(tokenizer_path)
            self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
            # infer max_seq_len from model input shape
            self.max_seq_len = self.model.input_shape[1] + 1
        except Exception as e:
            raise RuntimeError(f"Could not load model or tokenizer. Have you trained it yet? Error: {e}")

    def predict_next_word(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Predict the next word(s) for a given input text."""
        if not text:
            raise ValueError("Input text cannot be empty.")
            
        text_clean = re.sub(r"[^a-z\s]", "", text.lower().strip())
        seq = self.tokenizer.texts_to_sequences([text_clean])[0]
        seq = pad_sequences([seq], maxlen=self.max_seq_len-1, padding='pre')
        
        probs = self.model.predict(seq, verbose=0)[0]
        top_idx = probs.argsort()[-top_n:][::-1]
        
        return [(self.index_word.get(i, '<unk>'), float(probs[i])) for i in top_idx]

    def auto_complete(self, seed: str, n_words: int = 5) -> str:
        """Auto-complete a sentence by predicting n words ahead."""
        result = seed
        for _ in range(n_words):
            next_w = self.predict_next_word(result, top_n=1)[0][0]
            result += ' ' + next_w
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Next Word Predictor")
    parser.add_argument("text", type=str, help="Input text to predict next word for.")
    parser.add_argument("--top_n", type=int, default=3, help="Number of predictions to return.")
    parser.add_argument("--complete", type=int, default=0, help="Auto-complete n words.")
    args = parser.parse_args()

    predictor = NextWordPredictor()
    
    print(f"\nInput: '{args.text}'")
    
    if args.complete > 0:
        completed = predictor.auto_complete(args.text, n_words=args.complete)
        print(f"Auto-completed ({args.complete} words): '{completed}'")
    else:
        predictions = predictor.predict_next_word(args.text, top_n=args.top_n)
        print("Predictions:")
        for word, prob in predictions:
            print(f"  - {word}: {prob*100:.2f}%")
