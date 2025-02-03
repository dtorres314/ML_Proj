import os
import joblib
from src.extract_data import extract_relevant_info


def predict_labels(file_path, model_dir):
    """
    Predict Section ID for a given XML file using the trained model.

    Args:
        file_path (str): Path to the XML file.
        model_dir (str): Directory containing the trained model and vectorizer.

    Returns:
        dict: A dictionary containing the Problem ID, predicted Section ID, and related details.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Extract information from the XML file
    extracted_data = extract_relevant_info(file_path)
    problem_id = extracted_data.get("ProblemID", "")
    statement = extracted_data.get("Statement", "")

    # Vectorize the statement
    input_vector = vectorizer.transform([statement]).toarray()

    # Predict Section ID
    predicted_section_id = model.predict(input_vector)[0]

    # Construct the prediction result
    return {
        "ProblemID": problem_id,
        "PredictedSectionID": predicted_section_id,
    }
