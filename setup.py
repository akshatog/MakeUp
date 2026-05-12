from setuptools import setup, find_packages

setup(
    name="next_word_prediction",
    version="0.1.0",
    description="Next Word Prediction using NLP & LSTM",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "pandas",
        "matplotlib",
        "nltk",
        "wordcloud"
    ],
    python_requires=">=3.8",
)
