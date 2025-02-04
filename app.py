import os
import traceback
from flask import Flask, render_template, request, jsonify

from src.extract_data import extract_relevant_info
from src.train_model import train_model_pipeline
from src.predict_model import predict_labels

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
    Load and list all .xml files from the data directory.
    """
    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.lower().endswith(".xml"):
                # Normalize path (Windows backslash to forward slash)
                rel_path = os.path.relpath(os.path.join(root, filename), DATA_DIR)
                rel_path = rel_path.replace("\\", "/")
                files.append(rel_path)

    if not files:
        return jsonify({"files": [], "message": "No XML files found in data directory."})

    return jsonify({"files": files})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Preprocess each XML file by extracting text and saving it to outputs/<same structure>.txt
    """
    try:
        if not request.is_json:
            return jsonify({"message": "Expected JSON body"}), 415

        payload = request.json
        files = payload.get("files", [])
        if not files:
            return jsonify({"results": [], "message": "No files selected."}), 400

        results = []
        for rel_path in files:
            # Ensure consistent path
            norm_path = rel_path.replace("\\", "/")
            input_path = os.path.join(DATA_DIR, norm_path)

            if not os.path.exists(input_path):
                results.append({
                    "file": norm_path,
                    "status": "Error: File not found"
                })
                continue

            if not norm_path.lower().endswith(".xml"):
                results.append({
                    "file": norm_path,
                    "status": "Error: Not an XML file"
                })
                continue

            # Build the output path in outputs, with same structure
            # e.g. outputs/<book_id>/<chapter_id>/<section_id>/<problem>.txt
            output_full = os.path.join(OUTPUT_DIR, norm_path)
            # Replace .xml with .txt
            base_no_ext = os.path.splitext(output_full)[0]
            output_txt = base_no_ext + ".txt"

            try:
                text_content = extract_relevant_info(input_path)
                os.makedirs(os.path.dirname(output_txt), exist_ok=True)

                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(text_content)

                results.append({
                    "file": norm_path,
                    "status": "Processed",
                    "output": output_txt
                })
            except Exception as e:
                error_message = traceback.format_exc()
                results.append({
                    "file": norm_path,
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
    Train the ML model on all .txt files under outputs/ folder.
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
    Upload a single XML file into the data/ folder (top level).
    """
    if "file" not in request.files:
        return jsonify({"message": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file."}), 400

    if not file.filename.lower().endswith(".xml"):
        return jsonify({"message": "Only XML files allowed."}), 400

    save_path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    return jsonify({"message": "File uploaded", "file": file.filename})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the Book, Chapter, Section ID for a single XML file using the trained model.
    """
    if not request.is_json:
        return jsonify({"message": "Expected JSON"}), 415

    data = request.json
    file_rel = data.get("file", "")
    if not file_rel:
        return jsonify({"message": "No file provided"}), 400

    # Rebuild input path
    norm_path = file_rel.replace("\\", "/")
    input_path = os.path.join(DATA_DIR, norm_path)

    try:
        pred = predict_labels(input_path, MODEL_DIR)
        return jsonify({
            "file": file_rel,
            "book_id": pred["book_id"],
            "chapter_id": pred["chapter_id"],
            "section_id": pred["section_id"]
        })
    except Exception as e:
        err_trace = traceback.format_exc()
        return jsonify({
            "file": file_rel,
            "error": str(e),
            "details": err_trace
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
