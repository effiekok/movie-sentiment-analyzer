# 🎬 Movie Review Sentiment Analyzer

An interactive web application built with **Streamlit** that performs sentiment analysis on movie reviews. The project trains and compares two different machine learning approaches on the classic IMDB dataset:

1. **SVM (Support Vector Machine):** Utilizing `TfidfVectorizer` for text vectorization and `LinearSVC` for classification.
2. **Neural Network:** A Deep Learning model built with **TensorFlow/Keras** using an `Embedding` layer and `GlobalAveragePooling1D`.

## 🚀 Features
* **Live Model Training:** Trains both models on the fly and displays live training validation.
* **Performance Metrics:** Side-by-side accuracy metrics and dynamic classification reports for both models.
* **Interactive Testing:** Type in any custom movie review to see real-time predictions and confidence scores from both architectures.

## 🛠️ Tech Stack
* Python
* Streamlit
* Scikit-Learn
* TensorFlow / Keras
* Pandas & NumPy
