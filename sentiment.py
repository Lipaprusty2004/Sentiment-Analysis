import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎭 Sentiment Analysis using Logistic Regression")

# 1. Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"c:\Users\LIPA\Downloads\combined_emotion.csv")
    return df

df = load_data()

# 2. Train-test split
X = df['sentence']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build model pipeline
@st.cache_resource
def train_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('logreg', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model()

# 4. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show metrics
st.subheader("📊 Model Evaluation")
st.text(f"Accuracy: {accuracy * 100:.2f}%")

if st.checkbox("Show classification report"):
    st.text(classification_report(y_test, y_pred))

# 5. User input for prediction
st.subheader("✍️ Enter a sentence to predict emotion")
user_input = st.text_input("Your sentence here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"Predicted Emotion: **{prediction}**")
