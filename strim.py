import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load and train
df = pd.read_csv(r"C:\Users\LIPA\Downloads\spam_ham_dataset.csv (1).zip")
X = df['text']
y = df['label_num']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# UI
st.title("📧 Spam Message Detector")
st.write("Enter one or more messages below (each on a new line):")

user_input = st.text_area("Your Messages", height=250)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("✅ Predict"):
        messages = [line.strip() for line in user_input.strip().split("\n") if line.strip()]
        if not messages:
            st.warning("Please enter some messages.")
        else:
            vectors = vectorizer.transform(messages)
            predictions = model.predict(vectors)
            probs = model.predict_proba(vectors)

            st.markdown("### 📢 Prediction Results:")
            for i, msg in enumerate(messages):
                label = "🔴 SPAM" if predictions[i] == 1 else "🟢 NOT SPAM"
                confidence = probs[i][predictions[i]] * 100
                st.markdown(f"**Message {i+1}:** {msg}\n\n➡️ **{label}** (Confidence: {confidence:.2f}%)")

with col2:
    if st.button("❌ Exit"):
        st.warning("Session stopped. You can close the browser tab.")
        st.stop()  # stops the app from continuing further
