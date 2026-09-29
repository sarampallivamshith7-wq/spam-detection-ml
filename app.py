from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

EXAMPLES = {
    "Prize message": "Congratulations! You have won a free prize! Call 09061234567 now to claim.",
    "Urgent offer": "URGENT! Your mobile number has been selected for a cash reward. Text CLAIM to 80082 to collect.",
    "Casual chat": "Hey, are we still meeting for lunch tomorrow at 1?",
    "Reminder": "Can you pick up some milk on your way home? Thanks!",
}

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the saved model and vectorizer once (no retraining)."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def set_example(text: str) -> None:
    st.session_state["message"] = text


# ---------- Load model ----------
try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `spam_model.pkl` and/or `tfidf_vectorizer.pkl`. "
        "Place both files in the same folder as `app.py` and restart the app."
    )
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** Linear SVM")
    st.write("**Feature Extraction:** TF-IDF")
    st.subheader("Evaluation Metrics")
    st.metric("Accuracy", "98.07%")
    st.metric("Spam Precision", "99%")
    st.metric("Spam Recall", "85%")
    st.metric("Spam F1-score", "92%")
    st.caption("Metrics were measured on the held-out test set during model development.")

# ---------- Main page ----------
st.title("📩 SMS Spam Detector")
st.write(
    "A machine learning app that classifies SMS messages as spam or not spam, "
    "using TF-IDF features and a Linear SVM trained on the SMS Spam Collection dataset."
)

st.subheader("Try an example")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(text,), use_container_width=True)

st.subheader("Enter a message")
message = st.text_area(
    "Message", key="message", height=150,
    placeholder="Type or paste an SMS message here...",
    label_visibility="collapsed",
)

if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message before clicking Check Message.")
    else:
        features = vectorizer.transform([message])
        prediction = int(model.predict(features)[0])  # 0 = Ham, 1 = Spam

        st.subheader("Result")
        if prediction == 1:
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Not Spam")
