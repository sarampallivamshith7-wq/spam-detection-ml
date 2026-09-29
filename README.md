# 📧 Spam Detection using Machine Learning

A machine learning-based SMS spam detection system that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and machine learning.

The project uses **TF-IDF** for text feature extraction and compares multiple classification algorithms before selecting a **Linear Support Vector Machine (Linear SVM)** as the final model.

---

## 🚀 Features

- SMS text preprocessing
- Exploratory Data Analysis (EDA)
- TF-IDF feature extraction
- Multiple machine learning models
- Model performance comparison
- Linear SVM hyperparameter tuning
- Spam/Ham classification
- Saved ML model and TF-IDF vectorizer
- Interactive Streamlit web application

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**.

The dataset contains SMS messages labeled as:

- `ham` → legitimate message
- `spam` → spam message

The original dataset contains approximately **5,500 SMS messages**.

Dataset source:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 🧠 Machine Learning Pipeline

```text
SMS Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train/Test Split
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Model Training
     │
     ├── Multinomial Naive Bayes
     │
     ├── Logistic Regression
     │
     └── Linear SVM
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final Linear SVM Model
     │
     ▼
Streamlit Application
