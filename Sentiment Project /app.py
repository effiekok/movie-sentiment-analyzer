import streamlit as st
import pickle
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[import]

MAX_LENGTH  = 200

# Text Cleaner Helper
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Config browser layout
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="centered")
st.title("Movie Review Sentiment Analyzer")
st.caption("Compares two pre-trained ML models: **SVM + TF-IDF** vs. a **Neural Network**.")

# Load models from files
@st.cache_resource
def load_pre_trained_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    nn = tf.keras.models.load_model('nn_model.keras')
    return tfidf, svm, tokenizer, nn

# Safe file checking
required_files = ['tfidf_vectorizer.pkl', 'svm_model.pkl', 'tokenizer.pkl', 'nn_model.keras']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing model files on GitHub: {missing_files}")
    st.stop()

tfidf, svm, tokenizer, nn = load_pre_trained_models()
st.success("Pre-trained models loaded instantly!")

# Model Performance Display 
st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("SVM (TF-IDF) Accuracy", "88.4%")
c2.metric("Neural Network Accuracy", "87.5%")

st.divider()

# Prediction UI 
st.subheader("Try it out")
review = st.text_area("Enter a movie review in English:", placeholder="e.g. This movie was fantastic!", height=140)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter text first.")
    else:
        cleaned = clean_text(review)
        
        # SVM Prediction
        pred_svm = int(svm.predict(tfidf.transform([cleaned]))[0])
        
        # NN Prediction
        seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LENGTH, padding="post", truncating="post")
        prob_nn = float(nn.predict(seq, verbose=0)[0][0])
        pred_nn = int(prob_nn > 0.5)

        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Neural Network**")
            nn_label = "Positive" if pred_nn else "Negative"
            nn_conf = prob_nn if pred_nn else 1 - prob_nn
            st.success(nn_label) if pred_nn else st.error(nn_label)
            st.progress(nn_conf, text=f"Confidence: {nn_conf:.0%}")

        with col2:
            st.markdown("**SVM (TF-IDF)**")
            svm_label = "Positive" if pred_svm else "Negative"
            st.success(svm_label) if pred_svm else st.error(svm_label)