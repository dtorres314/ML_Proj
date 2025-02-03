import os
import joblib
from src.extract_data import extract_relevant_info

def predict_labels(file_path, model_dir):
    """
    Predict 'bookID-chapterID-sectionID' for a single XML file using the trained model.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Extract textual content from XML
    extracted_text = extract_relevant_info(file_path)
    if not extracted_text.strip():
        raise ValueError("No meaningful text extracted from the XML file.")

    # Vectorize and predict
    input_vec = vectorizer.transform([extracted_text]).toarray()
    predicted_label = model.predict(input_vec)[0]  # 'bookID-chapterID-sectionID'

    parts = predicted_label.split("-")
    if len(parts) == 3:
        book_id, chapter_id, section_id = parts
    else:
        book_id, chapter_id, section_id = ("Unknown", "Unknown", "Unknown")

    return {
        "book_id": book_id,
        "chapter_id": chapter_id,
        "section_id": section_id
    }
