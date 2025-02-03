from flask import Flask, render_template, request, jsonify
from src.extract_data import extract_relevant_info
from src.train_model import train_model_pipeline
from src.predict_model import predict_labels
import os
import traceback

app = Flask(__name__)

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "model"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route("/")
def index():
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
                rel_path = os.path.relpath(os.path.join(root, filename), DATA_DIR)
                files.append(rel_path)

    if not files:
        return jsonify({"files": [], "message": "No XML files found in the data directory."})

    return jsonify({"files": files})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Preprocess (extract text) from each XML file and save it to a .txt file in
    outputs/<book_id>/<chapter_id>/<section_id> directory.
    """
    try:
        if not request.is_json:
            return jsonify({"message": "Invalid request format. Expected JSON."}), 415

        files = request.json.get("files", [])
        if not files:
            return jsonify({"results": [], "message": "No files selected for processing."}), 400

        results = []
        for rel_path in files:
            # Convert backslashes to forward slashes (Windows compatibility)
            norm_path = rel_path.replace("\\", "/")
            input_path = os.path.join(DATA_DIR, norm_path)

            # Split the path by "/"
            path_parts = norm_path.split("/")

            # We want the last 4 parts: [book_id, chapter_id, section_id, problemName.xml]
            if len(path_parts) < 4:
                results.append({
                    "file": rel_path,
                    "status": "Error: Directory structure not recognized (needs at least 4 parts)."
                })
                continue

            # Extract the last four parts
            book_id, chapter_id, section_id, xml_file = path_parts[-4:]
            problem_name = os.path.splitext(xml_file)[0]

            # Output path: outputs/<book_id>/<chapter_id>/<section_id>/<problem>.txt
            out_dir = os.path.join(OUTPUT_DIR, book_id, chapter_id, section_id)
            output_file = os.path.join(out_dir, f"{problem_name}.txt")

            try:
                if not os.path.exists(input_path):
                    results.append({"file": rel_path, "status": "Error: File not found"})
                    continue

                text_content = extract_relevant_info(input_path)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

                results.append({"file": rel_path, "status": "Processed", "output": output_file})
            except Exception as e:
                error_message = traceback.format_exc()
                results.append({
                    "file": rel_path,
                    "status": f"Error: {str(e)}",
                    "details": error_message
                })

        return jsonify({"results": results})

    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"message": f"An error occurred: {str(e)}", "details": error_message}), 500


@app.route("/train_model", methods=["POST"])
def train_model():
    """
    Train a supervised learning model using the preprocessed text files.
    """
    try:
        training_results = train_model_pipeline(OUTPUT_DIR, MODEL_DIR)
        return jsonify({"status": "success", "training_results": training_results})
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "details": error_message}), 500


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
    Predict the Book, Chapter, and Section ID for the given XML file.
    """
    if not request.is_json:
        return jsonify({"message": "Invalid request format. Expected JSON."}), 415

    file = request.json.get("file", "")
    if not file:
        return jsonify({"message": "No file selected for prediction."}), 400

    input_path = os.path.join(DATA_DIR, file)
    try:
        prediction = predict_labels(input_path, MODEL_DIR)
        # Return the separate fields
        return jsonify({
            "file": file,
            "book_id": prediction["book_id"],
            "chapter_id": prediction["chapter_id"],
            "section_id": prediction["section_id"]
        })
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"file": file, "error": str(e), "details": error_message}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
