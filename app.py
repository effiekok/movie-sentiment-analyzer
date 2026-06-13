import streamlit as st
import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# Config 
FILENAME    = "IMDB Dataset.csv"
VOCAB_SIZE  = 10_000
MAX_LENGTH  = 200
EMBEDDING   = 16
EPOCHS      = 3

# Helpers 
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Model training 
@st.cache_resource(show_spinner=False)
def load_and_train():
    # Load dataset
    try:
        df = pd.read_csv(FILENAME, encoding="latin-1")
    except Exception:
        df = pd.read_csv(FILENAME, encoding="utf-8", encoding_errors="replace")

    # Normalise sentiment column
    for alias in ("label", "type"):
        if "sentiment" not in df.columns and alias in df.columns:
            df.rename(columns={alias: "sentiment"}, inplace=True)

    if "sentiment" not in df.columns:
        st.error("Dataset does not contain a sentiment column.")
        st.stop()

    if "unsup" in df["sentiment"].values:
        df = df[df["sentiment"] != "unsup"]

    df["label"]        = df["sentiment"].apply(lambda x: 1 if "pos" in str(x).lower() else 0)
    df["clean_review"] = df["review"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2, random_state=42
    )

    # SVM
    tfidf = TfidfVectorizer(max_features=5000)
    X_tr_tfidf = tfidf.fit_transform(X_train)
    X_te_tfidf = tfidf.transform(X_test)

    svm = LinearSVC(dual="auto")
    svm.fit(X_tr_tfidf, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_te_tfidf))
    svm_report = classification_report(
        y_test, svm.predict(X_te_tfidf),
        target_names=["Negative", "Positive"], output_dict=True
    )

    # Neural Network
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_tr_pad = pad_sequences(tokenizer.texts_to_sequences(X_train),
                             maxlen=MAX_LENGTH, padding="post", truncating="post")
    X_te_pad = pad_sequences(tokenizer.texts_to_sequences(X_test),
                             maxlen=MAX_LENGTH, padding="post", truncating="post")

    nn = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING, input_length=MAX_LENGTH),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    nn.fit(X_tr_pad, y_train, epochs=EPOCHS,
           validation_data=(X_te_pad, y_test), verbose=0)

    y_pred_nn = (nn.predict(X_te_pad, verbose=0) > 0.5).astype(int)
    nn_acc    = accuracy_score(y_test, y_pred_nn)
    nn_report = classification_report(
        y_test, y_pred_nn,
        target_names=["Negative", "Positive"], output_dict=True
    )

    return tfidf, svm, svm_acc, svm_report, tokenizer, nn, nn_acc, nn_report


def predict(text: str, tfidf, svm, tokenizer, nn):
    cleaned  = clean_text(text)
    pred_svm = int(svm.predict(tfidf.transform([cleaned]))[0])

    seq     = pad_sequences(tokenizer.texts_to_sequences([cleaned]),
                            maxlen=MAX_LENGTH, padding="post", truncating="post")
    prob_nn = float(nn.predict(seq, verbose=0)[0][0])
    pred_nn = int(prob_nn > 0.5)

    return pred_svm, pred_nn, prob_nn


# Page layout 
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("Compares two ML models — **SVM + TF-IDF** vs. a **Neural Network** — on the IMDB dataset.")

if not os.path.exists(FILENAME):
    st.error(f"Dataset file `{FILENAME}` not found. Place it in the same folder as `app.py`.")
    st.stop()

# Training
with st.spinner("Training models on IMDB dataset… (first run only, takes ~1 min)"):
    tfidf, svm, svm_acc, svm_report, tokenizer, nn, nn_acc, nn_report = load_and_train()

st.success("Models ready!")

#  Accuracy cards 
st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("SVM (TF-IDF) Accuracy",    f"{svm_acc:.1%}")
c2.metric("Neural Network Accuracy",  f"{nn_acc:.1%}")

with st.expander("Detailed classification report"):
    tab1, tab2 = st.tabs(["SVM", "Neural Network"])
    with tab1:
        st.dataframe(
            pd.DataFrame(svm_report).transpose().style.format("{:.2f}"),
            use_container_width=True,
        )
    with tab2:
        st.dataframe(
            pd.DataFrame(nn_report).transpose().style.format("{:.2f}"),
            use_container_width=True,
        )

st.divider()

# Prediction UI 
st.subheader("Try it out")
review = st.text_area(
    "Enter a movie review in English:",
    placeholder="e.g. This movie was absolutely fantastic! Great acting and a gripping story.",
    height=140,
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before clicking Analyze.")
    else:
        pred_svm, pred_nn, prob_nn = predict(review, tfidf, svm, tokenizer, nn)

        st.subheader("Results")
        col1, col2 = st.columns(2)

        # Neural Network result
        with col1:
            st.markdown("**Neural Network**")
            nn_label      = "Positive" if pred_nn else "Negative"
            nn_confidence = prob_nn if pred_nn else 1 - prob_nn
            if pred_nn:
                st.success(f"{nn_label}")
            else:
                st.error(f"{nn_label}")
            st.progress(nn_confidence, text=f"Confidence: {nn_confidence:.0%}")

        # SVM result
        with col2:
            st.markdown("**SVM (TF-IDF)**")
            svm_label = "Positive" if pred_svm else "Negative"
            if pred_svm:
                st.success(f"{svm_label}")
            else:
                st.error(f"{svm_label}")

        # Agreement banner
        st.divider()
        if pred_svm == pred_nn:
            st.info("Both models agree on the prediction.")
        else:
            st.warning(
                "The models disagree — the review may be ambiguous or mixed.")
