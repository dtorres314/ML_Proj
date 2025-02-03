import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_preprocessed_data(output_dir):
    """
    Gathers all .txt files from outputs/<book_id>/<chapter_id>/<section_id>.
    The label is 'bookID-chapterID-sectionID' derived from the folder structure.
    """
    rows = []
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, fname), output_dir)
                path_parts = rel_path.replace("\\", "/").split("/")
                if len(path_parts) >= 4:
                    # e.g. [book_id, chapter_id, section_id, problem.txt]
                    book_id, chapter_id, section_id = path_parts[:3]
                    label = f"{book_id}-{chapter_id}-{section_id}"
                else:
                    # If structure isn't 4 deep, skip or handle differently
                    continue

                with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                    text_content = f.read().strip()

                rows.append({"content": text_content, "label": label})
    return pd.DataFrame(rows)

def train_model_pipeline(output_dir, model_dir):
    df = load_preprocessed_data(output_dir)
    if df.empty:
        return {"error": "No preprocessed text files found. Please run Preprocess first."}

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["content"]).toarray()
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    return {
        "accuracy": report["accuracy"],
        "classification_report": report
    }
