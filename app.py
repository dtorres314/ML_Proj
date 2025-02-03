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
    Load and list all .xml files in the data directory.
    """
    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.endswith(".xml"):
                rel_path = os.path.relpath(os.path.join(root, filename), DATA_DIR)
                files.append(rel_path.replace("\\", "/"))
    if not files:
        return jsonify({"files": [], "message": "No XML files found."})
    return jsonify({"files": files})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Preprocess (extract text) from XML files and save to:
      outputs/<book_id>/<chapter_id>/<section_id>/<problem>.txt
    """
    try:
        if not request.is_json:
            return jsonify({"message": "Expected JSON payload"}), 415

        payload = request.json
        files = payload.get("files", [])
        if not files:
            return jsonify({"results": [], "message": "No files selected"}), 400

        results = []
        for rel_path in files:
            # Normalize path
            norm_path = rel_path.replace("\\", "/")
            input_path = os.path.join(DATA_DIR, norm_path)

            # Extract last 4 parts [book, chapter, section, problem.xml]
            path_parts = norm_path.split("/")
            if len(path_parts) < 4:
                results.append({
                    "file": norm_path,
                    "status": "Error: Not enough path segments (need >=4)."
                })
                continue

            book_id, chapter_id, section_id, problem_file = path_parts[-4:]
            problem_name, _ = os.path.splitext(problem_file)

            output_dir = os.path.join(OUTPUT_DIR, book_id, chapter_id, section_id)
            output_file = os.path.join(output_dir, f"{problem_name}.txt")

            if not os.path.exists(input_path):
                results.append({"file": rel_path, "status": "Error: File not found"})
                continue

            try:
                text_content = extract_relevant_info(input_path)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

                results.append({
                    "file": rel_path,
                    "status": "Processed",
                    "output": output_file
                })
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
        return jsonify({"message": str(e), "details": error_message}), 500


@app.route("/train_model", methods=["POST"])
def train_model():
    """
    Train a supervised learning model using the extracted .txt files
    that are stored in outputs/<book_id>/<chapter_id>/<section_id> subfolders.
    """
    try:
        result = train_model_pipeline(OUTPUT_DIR, MODEL_DIR)
        return jsonify({"status": "success", "training_results": result})
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "details": error_message}), 500


@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    Upload a single XML file from user. (Manually places it in data/<filename>)
    """
    if "file" not in request.files:
        return jsonify({"message": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    if not file.filename.endswith(".xml"):
        return jsonify({"message": "Only XML files allowed"}), 400

    save_path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    return jsonify({"message": "File uploaded", "file": file.filename})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict BookID, ChapterID, SectionID for single XML file using the trained model.
    """
    if not request.is_json:
        return jsonify({"message": "Expected JSON payload"}), 415

    req_data = request.json
    file = req_data.get("file", "")
    if not file:
        return jsonify({"message": "No file given"}), 400

    input_path = os.path.join(DATA_DIR, file.replace("\\", "/"))
    try:
        pred = predict_labels(input_path, MODEL_DIR)
        return jsonify({
            "file": file,
            "book_id": pred["book_id"],
            "chapter_id": pred["chapter_id"],
            "section_id": pred["section_id"]
        })
    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({
            "file": file,
            "error": str(e),
            "details": error_message
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
