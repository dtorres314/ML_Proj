from flask import Flask, render_template, request, jsonify
from src.extract_data import extract_relevant_info
from src.train_model import train_model_pipeline
from src.predict_model import predict_labels
import os
import traceback

app = Flask(__name__)

# Define constants for directories
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "model"

# Ensure the outputs and model directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


@app.route("/")
def index():
    """
    Render the main page.
    """
    return render_template("index.html")


@app.route("/load_files", methods=["POST"])
def load_files():
    """
    Load and list all XML files from the data directory.
    """
    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.endswith(".xml"):
                relative_path = os.path.relpath(os.path.join(root, filename), DATA_DIR)
                files.append(relative_path)

    if not files:
        return jsonify({"files": [], "message": "No XML files found in the data directory."})

    return jsonify({"files": files})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Extract relevant information from XML files and save it as .txt files.
    """
    try:
        # Ensure the request contains JSON data
        if not request.is_json:
            return jsonify({"message": "Invalid request format. Expected JSON."}), 415

        files = request.json.get("files", [])
        if not files:
            return jsonify({"results": [], "message": "No files selected for processing."}), 400

        results = []
        for file in files:
            input_path = os.path.join(DATA_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}.txt")

            try:
                # Check if the file exists
                if not os.path.exists(input_path):
                    results.append({"file": file, "status": "Error: File not found"})
                    continue

                # Extract relevant text
                extracted_data = extract_relevant_info(input_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the extracted text to a .txt file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_data)

                results.append({"file": file, "status": "Processed", "output": output_path})
            except Exception as e:
                # Handle and log errors
                error_message = traceback.format_exc()
                results.append({"file": file, "status": f"Error: {str(e)}", "details": error_message})

        return jsonify({"results": results})
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"message": f"An unexpected error occurred: {str(e)}", "details": error_message}), 500


@app.route("/train_model", methods=["POST"])
def train_model():
    """
    Train the supervised learning model using the preprocessed .txt files.
    """
    try:
        training_results = train_model_pipeline(OUTPUT_DIR, MODEL_DIR)
        return jsonify({"status": "success", "details": training_results})
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "details": error_message})


@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    Upload a new XML file from the user.
    """
    if "file" not in request.files:
        return jsonify({"message": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file."}), 400

    if file and file.filename.endswith(".xml"):
        save_path = os.path.join(DATA_DIR, file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        return jsonify({"message": "File uploaded successfully.", "file": file.filename})

    return jsonify({"message": "Only XML files are allowed."}), 400


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Section ID for the uploaded XML file using the trained model.
    """
    file = request.json.get("file", None)
    if not file:
        return jsonify({"message": "No file selected for prediction."}), 400

    input_path = os.path.join(DATA_DIR, file)
    try:
        prediction = predict_labels(input_path, MODEL_DIR)

        # Format the prediction result
        prediction_result = {
            "ProblemID": prediction.get("ProblemID"),
            "PredictedSectionID": prediction.get("PredictedSectionID"),
            "Book": f"Book 1",  # Example mapping for demonstration
            "Chapter": f"Chapter 5",  # Example mapping for demonstration
            "Section": f"Section {prediction.get('PredictedSectionID')}",
        }
        return jsonify(prediction_result)
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"file": file, "error": str(e), "details": error_message}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
