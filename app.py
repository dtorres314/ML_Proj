from flask import Flask, render_template, request, jsonify
import os
import traceback
import csv

from src.db_manager import init_db, insert_problem_entry
from src.extract_data import extract_relevant_info
from src.train_and_test import train_and_test_pipeline
from src.predict_model import predict_labels

app = Flask(__name__)

DATA_DIR = "data"
MODEL_DIR = "model"


# Store a global lookup_map (bookID, chapterID, sectionID) -> (bookName, chapterName, sectionName)
lookup_map = {}


# Initialize DB
init_db()

def load_contentbackup_csv():
    csv_path = os.path.join(os.path.dirname(__file__), "contentbackup.csv")
    if not os.path.exists(csv_path):
        print("WARNING: contentbackup.csv not found. Name lookup won't work.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Clean the fieldnames to remove leading/trailing spaces
        cleaned_fields = [fn.strip() for fn in reader.fieldnames]
        reader.fieldnames = cleaned_fields

        for row in reader:
            b_id = row["BookID"].strip()
            c_id = row["ChapterID"].strip()
            s_id = row["SectionID"].strip()

            b_name = row["BookName"].strip()
            c_name = row["ChapterName"].strip()
            sec_name = row["SectionName"].strip()

            key = (b_id, c_id, s_id)
            lookup_map[key] = (b_name, c_name, sec_name)

# Load CSV once on startup
load_contentbackup_csv()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_files", methods=["POST"])
def load_files():
    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.lower().endswith(".xml"):
                rel_path = os.path.relpath(os.path.join(root, filename), DATA_DIR)
                rel_path = rel_path.replace("\\", "/")
                files.append(rel_path)
    return jsonify({"files": files})


@app.route("/extract_and_save_db", methods=["POST"])
def extract_and_save_db():
    """
    For each file path, parse out Book/Chapter/Section from the first 3 path segments,
    then the 4th path segment is the "problem folder",
    and the 5th is the .xml file (problemFile).
    
    Path example: "1/1/2/1288/1288.xml"
    => bookId = '1', chapterId = '1', sectionId = '2'
       problemFolder = '1288', problemFile = '1288.xml'
       problemId = '1288'
    """
    try:
        if not request.is_json:
            return jsonify({"message": "Expected JSON body"}), 415

        data = request.json
        files = data.get("files", [])
        if not files:
            return jsonify({"results": [], "message": "No files specified"}), 400

        results = []
        for rel_path in files:
            # e.g. "1/1/2/1288/1288.xml"
            norm_path = rel_path.replace("\\", "/")
            full_path = os.path.join(DATA_DIR, norm_path)

            path_parts = norm_path.split("/")
            if len(path_parts) != 5:
                results.append({
                    "file": rel_path,
                    "status": "Error: path must have exactly 5 parts (book/chapter/section/problemFolder/problemFile.xml)."
                })
                continue

            # Extract the fields
            book_id = path_parts[0]       # '1'
            chapter_id = path_parts[1]    # '1'
            section_id = path_parts[2]    # '2'
            problem_folder = path_parts[3]   # '1288'
            problem_file = path_parts[4]     # '1288.xml'

            # Problem ID from the file name
            problem_id = os.path.splitext(problem_file)[0]  # '1288'

            if not os.path.exists(full_path):
                results.append({
                    "file": rel_path,
                    "status": "Error: File not found"
                })
                continue

            try:
                # Extract text content from .xml
                text_content = extract_relevant_info(full_path)
                insert_problem_entry(
                    problem_id=problem_id,
                    book_id=book_id,
                    chapter_id=chapter_id,
                    section_id=section_id,
                    content=text_content
                )
                results.append({
                    "file": rel_path,
                    "status": "Saved to DB",
                    "bookId": book_id,
                    "chapterId": chapter_id,
                    "sectionId": section_id,
                    "problemFolder": problem_folder,
                    "problemId": problem_id
                })
            except Exception as e:
                err_message = traceback.format_exc()
                results.append({
                    "file": rel_path,
                    "status": f"Error: {str(e)}",
                    "details": err_message
                })

        return jsonify({"results": results})

    except Exception as e:
        err = traceback.format_exc()
        return jsonify({"message": str(e), "details": err}), 500

@app.route("/train_and_test", methods=["POST"])
def train_and_test():
    """
    Splits data 70/30, trains, logs test results in test_summary_table,
    saves model to 'model/' folder.
    Returns summary stats (accuracy, etc).
    """
    try:
        summary = train_and_test_pipeline(MODEL_DIR)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        err = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "details": err}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"message": "Expected JSON"}), 415

    body = request.json
    rel_file = body.get("file", "")
    if not rel_file:
        return jsonify({"message": "No file provided"}), 400

    full_path = os.path.join(DATA_DIR, rel_file.replace("\\", "/"))
    try:
        # 1) Use the model to predict book_id, chapter_id, section_id
        preds = predict_labels(full_path, MODEL_DIR)
        b_id = preds["book_id"]
        c_id = preds["chapter_id"]
        s_id = preds["section_id"]

        # 2) Lookup names in contentbackup.csv
        b_name, c_name, sec_name = ("Unknown Book", "Unknown Chapter", "Unknown Section")
        key = (b_id, c_id, s_id)
        if key in lookup_map:
            b_name, c_name, sec_name = lookup_map[key]

        # 3) Return IDs and names
        return jsonify({
            "file": rel_file,
            "book_id": b_id,
            "book_name": b_name,
            "chapter_id": c_id,
            "chapter_name": c_name,
            "section_id": s_id,
            "section_name": sec_name
        })
    except Exception as e:
        err = traceback.format_exc()
        return jsonify({
            "file": rel_file,
            "error": str(e),
            "details": err
        }), 500


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"message": "No selected file."}), 400

    if not f.filename.lower().endswith(".xml"):
        return jsonify({"message": "Only XML files allowed"}), 400

    save_path = os.path.join(DATA_DIR, f.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f.save(save_path)

    return jsonify({"message": "File uploaded", "file": f.filename})


if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
