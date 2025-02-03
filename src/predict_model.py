import os
import joblib
from src.extract_data import extract_relevant_info

def predict_labels(file_path, model_dir):
    """
    Predict 'bookID-chapterID-sectionID' for the given XML file using the trained model.
    Returns a dict with separate fields.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Trained model or vectorizer not found. Please train first.")

    # Load
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Extract
    text = extract_relevant_info(file_path).strip()
    if not text:
        raise ValueError("No meaningful text found in XML for prediction.")

    X_vec = vectorizer.transform([text]).toarray()
    label = model.predict(X_vec)[0]  # 'bookID-chapterID-sectionID'

    parts = label.split("-")
    if len(parts) != 3:
        return {"book_id": "Unknown", "chapter_id": "Unknown", "section_id": "Unknown"}
    return {
        "book_id": parts[0],
        "chapter_id": parts[1],
        "section_id": parts[2]
    }
