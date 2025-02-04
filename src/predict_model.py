import os
import joblib
from src.extract_data import extract_relevant_info

def predict_labels(file_path, model_dir):
    """
    Predict book-chapter-section for a single XML using the model in model_dir
    """
    model_file = os.path.join(model_dir, "model.pkl")
    vec_file = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_file) or not os.path.exists(vec_file):
        raise FileNotFoundError("Model or vectorizer not found. Please train first.")

    clf = joblib.load(model_file)
    vectorizer = joblib.load(vec_file)

    # Extract text
    content = extract_relevant_info(file_path)
    if not content.strip():
        raise ValueError("No meaningful text in XML file.")

    # Vectorize
    X_vec = vectorizer.transform([content]).toarray()
    pred = clf.predict(X_vec)[0]  # '1-24-185'

    parts = pred.split("-")
    if len(parts) == 3:
        return {
            "book_id": parts[0],
            "chapter_id": parts[1],
            "section_id": parts[2]
        }
    else:
        return {
            "book_id": "Unknown",
            "chapter_id": "Unknown",
            "section_id": "Unknown"
        }
