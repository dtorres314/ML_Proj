import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_preprocessed_data(output_dir):
    """
    Build a dataset from the text files in `output_dir`.
    Each text file corresponds to a label with format 'bookID-chapterID-sectionID'.
    """
    data = []
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                # Label is derived from the folder structure: bookID-chapterID-sectionID
                rel_path = os.path.relpath(os.path.join(root, filename), output_dir)
                label_dirs = os.path.dirname(rel_path).split(os.sep)  # [bookID, chapterID, sectionID]
                if len(label_dirs) == 3:
                    label = "-".join(label_dirs)  # e.g. '1-24-185'
                else:
                    # If not exactly 3 levels, skip or handle differently
                    continue

                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                data.append({"content": content, "label": label})

    return pd.DataFrame(data)

def train_model_pipeline(output_dir, model_dir):
    df = load_preprocessed_data(output_dir)
    if df.empty:
        return {"error": "No training data found in outputs directory. Preprocess files first."}

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["content"]).toarray()
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model and vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    # Return a summary
    return {
        "accuracy": report["accuracy"],
        "classification_report": report
    }
