import pandas as pd     
import numpy as np      
import re               
import time             
import os               

# Machine Learning libraries (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Deep Learning libraries (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


# STEP 1: LOAD AND PREPARE DATA
print("\n--- 1. STARTING PROCESS ---")

FILENAME = 'IMDB Dataset.csv' 

if not os.path.exists(FILENAME):
    print(f"ERROR: File '{FILENAME}' not found!")
    exit()

print(f"Loading data from {FILENAME}...")

# Load with encoding fallback
try:
    df = pd.read_csv(FILENAME, encoding='latin-1')
except:
    df = pd.read_csv(FILENAME, encoding='utf-8', encoding_errors='replace')

print(f"Initial records: {len(df)}")

# Detect the correct sentiment column (sentiment/label/type)
if 'sentiment' not in df.columns and 'label' in df.columns:
    df.rename(columns={'label': 'sentiment'}, inplace=True)
if 'sentiment' not in df.columns and 'type' in df.columns:
    df.rename(columns={'type': 'sentiment'}, inplace=True)

if 'sentiment' not in df.columns:
    print("ERROR: No sentiment column found.")
    exit()

# Remove unlabeled 'unsup' entries
if 'unsup' in df['sentiment'].values:
    print("Removing unlabeled records (unsup)...")
    df = df[df['sentiment'] != 'unsup']

# Encode labels: Positive=1, Negative=0
df['label'] = df['sentiment'].apply(lambda x: 1 if 'pos' in str(x).lower() else 0)

print(f"Final records for training: {len(df)}")


# STEP 2: TEXT PREPROCESSING (CLEANING)
print("\n--- 2. CLEANING TEXTS ---")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)
print("Cleaning complete.")

# Split into Train (80%) and Test (20%)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42
)
print(f"Train set: {len(X_train_raw)}, Test set: {len(X_test_raw)}")


# STEP 3: MODEL 1 - SVM WITH TF-IDF
print("\n--- 3. TRAINING SVM WITH TF-IDF ---")

tfidf = TfidfVectorizer(max_features=5000)

print("Converting texts to TF-IDF vectors...")
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

svm_model = LinearSVC(dual='auto')

print("Training SVM...")
start_time = time.time()
svm_model.fit(X_train_tfidf, y_train)
svm_train_time = time.time() - start_time
print(f"-> SVM Training Time: {svm_train_time:.2f} seconds")

print("Evaluating SVM...")
start_time = time.time()
y_pred_svm = svm_model.predict(X_test_tfidf)
svm_inference_time = (time.time() - start_time) / len(X_test_raw)

print(classification_report(y_test, y_pred_svm, target_names=['Negative', 'Positive']))


# STEP 4: MODEL 2 - NEURAL NETWORK (NN)
print("\n--- 4. TRAINING NEURAL NETWORK ---")

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 16
EPOCHS = 3

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_raw)

X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
X_test_seq = tokenizer.texts_to_sequences(X_test_raw)

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Model Architecture
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training Neural Network...")
start_time = time.time()
history = model.fit(X_train_padded, y_train, epochs=EPOCHS, validation_data=(X_test_padded, y_test), verbose=1)
nn_train_time = time.time() - start_time

print("Evaluating Neural Network...")
start_time = time.time()
y_pred_nn = (model.predict(X_test_padded) > 0.5).astype(int)
nn_inference_time = (time.time() - start_time) / len(X_test_raw)

print(classification_report(y_test, y_pred_nn, target_names=['Negative', 'Positive']))


# STEP 5: COMPARISON & APPLICATION (WITH DESIGN & COLORS)
print("\n" + "="*60)
print(f"{'Model':<20} | {'Train Time':<15} | {'Avg Pred Time':<18} | {'Accuracy':<10}")
print("-" * 75)
svm_acc = accuracy_score(y_test, y_pred_svm)
nn_acc = accuracy_score(y_test, y_pred_nn)
print(f"{'SVM (TF-IDF)':<20} | {svm_train_time:<15.4f} | {svm_inference_time:<18.6f} | {svm_acc:<10.4f}")
print(f"{'Neural Network':<20} | {nn_train_time:<15.4f} | {nn_inference_time:<18.6f} | {nn_acc:<10.4f}")
print("="*60)


# --- STEP 6: SAVE WORKED MODELS FOR THE WEB APP ---
import pickle

print("\nSaving trained models locally...")
# Save SVM and TF-IDF Vectorizer
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save the Tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save Keras Neural Network
model.save('nn_model.keras')
print("All models successfully saved as files!")


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_box(text, color=Colors.CYAN):
    lines = text.split('\n')
    width = max(len(line) for line in lines) + 4
    print(color + "╔" + "═" * width + "╗")
    for line in lines:
        print(f"║  {line:<{width-4}}  ║")
    print("╚" + "═" * width + "╝" + Colors.END)

def predict_sentiment_app(text):
    # 1. Clean text
    cleaned = clean_text(text)

    # SVM Prediction
    pred_svm = svm_model.predict(tfidf.transform([cleaned]))[0]

    # NN Prediction
    seq_nn = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LENGTH, padding='post', truncating='post')
    prob_nn = model.predict(seq_nn, verbose=0)[0][0]

    # Format result
    sentiment_color = Colors.GREEN if prob_nn > 0.5 else Colors.RED
    sentiment_text = "POSITIVE" if prob_nn > 0.5 else "NEGATIVE"

    print(f"\n{Colors.BOLD}ANALYSIS RESULTS:{Colors.END}")
    print("-" * 30)
    print(f"Text: {Colors.BLUE}'{text}'{Colors.END}")
    print("-" * 30)
    print(f"{Colors.BOLD}Neural Network:{Colors.END} {sentiment_color}{sentiment_text}{Colors.END} (Probability: {prob_nn:.2f})")

    # SVM (for confirmation)
    svm_res = "POSITIVE" if pred_svm == 1 else "NEGATIVE"
    print(f"SVM Model:      {svm_res}")
    print("-" * 30)

# MAIN PROGRAM (INTERFACE)
print("\n")
print_box("  SENTIMENT ANALYSIS APPLICATION  \n     (Movie Review Sentiment)", Colors.HEADER)

while True:
    print(f"\n{Colors.CYAN}--- New Review ---{Colors.END}")
    print("Write your review in English (or 'exit' to quit):")

    # Input
    txt = input(f"{Colors.GREEN}  Type here: {Colors.END}")

    if txt.lower() == 'exit':
        print_box("Application terminated.\nThank you!", Colors.RED)
        break
    
    if txt.strip(): 
        predict_sentiment_app(txt)