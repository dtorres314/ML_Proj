import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_preprocessed_data(output_dir):
    data = []
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                label = os.path.basename(root)
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                data.append({"content": content, "label": label})
    return pd.DataFrame(data)


def train_model_pipeline(output_dir, model_dir):
    df = load_preprocessed_data(output_dir)

    if df.empty:
        raise ValueError("No preprocessed data found for training.")

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["content"]).toarray()
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    return report
