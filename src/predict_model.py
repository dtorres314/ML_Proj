import os
import joblib
from src.extract_data import extract_relevant_info

def predict_labels(file_path, model_dir):
    """
    Predict 'book_id-chapter_id-section_id' for a new XML file.
    Returns a dict with separate fields for book_id, chapter_id, and section_id.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    vec_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Model or vectorizer not found. Please train first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    text_content = extract_relevant_info(file_path)
    if not text_content.strip():
        raise ValueError("No meaningful text extracted from XML.")

    X_vec = vectorizer.transform([text_content]).toarray()
    label = model.predict(X_vec)[0]  # e.g. '1-24-185'

    parts = label.split("-")
    if len(parts) == 3:
        book_id, chapter_id, section_id = parts
    else:
        book_id = chapter_id = section_id = "Unknown"

    return {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "section_id": section_id
    }
