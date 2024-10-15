import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import re
import time

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)


# Define a function to preprocess queries (handling HTML tags, symbols, etc.)
def preprocess_query(query):
    # Lowercase the query
    query = query.lower()

    # Remove common XSS/SQLi characters that are not meaningful in classification
    query = re.sub(
        r'[<>;"\'/]', " ", query
    )  # Removing some HTML tags and SQL characters

    # Replace multiple spaces with single space
    query = re.sub(r"\s+", " ", query).strip()

    return query


# Train and save model for SQLi detection
def train_sqli_model():
    print("Training SQLi model... Please wait.")

    # Simulate loading (optional, to make loading more apparent)
    time.sleep(1)

    # Load SQLi training dataset
    sqli_train_data = pd.read_csv(
        "E:/AI DATASET/AI DATASET/SQL/train_data.csv"
    )  # Adjust path

    # Preprocess queries
    sqli_train_data["Query"] = sqli_train_data["Query"].apply(preprocess_query)

    # Initialize vectorizer and transform data
    sqli_vectorizer = CountVectorizer()
    X_sqli_train = sqli_vectorizer.fit_transform(sqli_train_data["Query"])
    y_sqli_train = sqli_train_data["Label"]

    # Remove NaN values
    non_nan_indices_sqli = ~y_sqli_train.isna()
    X_sqli_train = X_sqli_train[non_nan_indices_sqli]
    y_sqli_train = y_sqli_train[non_nan_indices_sqli]

    # Create and train a Random Forest model for SQLi
    sqli_model = RandomForestClassifier(n_estimators=100)
    sqli_model.fit(X_sqli_train, y_sqli_train)

    # Save the SQLi model
    with open("models/sqli_model.pkl", "wb") as file:
        pickle.dump(sqli_model, file)

    # Save the fitted vectorizer for SQLi
    with open("models/sqli_vectorizer.pkl", "wb") as file:
        pickle.dump(sqli_vectorizer, file)

    print("SQLi model and vectorizer saved! ✅")


# Train and save model for XSS detection
def train_xss_model():
    print("Training XSS model... Please wait.")

    # Simulate loading (optional, to make loading more apparent)
    time.sleep(1)

    # Load XSS training dataset
    xss_train_data = pd.read_csv(
        "E:/AI DATASET/AI DATASET/XSS/train_data.csv"
    )  # Adjust path

    # Preprocess queries
    xss_train_data["Sentence"] = xss_train_data["Sentence"].apply(preprocess_query)

    # Initialize vectorizer and transform data
    xss_vectorizer = CountVectorizer()
    X_xss_train = xss_vectorizer.fit_transform(xss_train_data["Sentence"])
    y_xss_train = xss_train_data["Label"]

    # Remove NaN values
    non_nan_indices_xss = ~y_xss_train.isna()
    X_xss_train = X_xss_train[non_nan_indices_xss]
    y_xss_train = y_xss_train[non_nan_indices_xss]

    # Create and train a Random Forest model for XSS
    xss_model = RandomForestClassifier(n_estimators=100)
    xss_model.fit(X_xss_train, y_xss_train)

    # Save the XSS model
    with open("models/xss_model.pkl", "wb") as file:
        pickle.dump(xss_model, file)

    # Save the fitted vectorizer for XSS
    with open("models/xss_vectorizer.pkl", "wb") as file:
        pickle.dump(xss_vectorizer, file)

    print("XSS model and vectorizer saved! ✅")


# Main training function
if __name__ == "__main__":
    print("Starting model training...")

    train_sqli_model()  # Train SQLi model
    train_xss_model()  # Train XSS model

    print("Training complete for both SQLi and XSS models! 🎉")
