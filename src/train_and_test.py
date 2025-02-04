import random
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from src.db_manager import fetch_training_data, clear_test_summary, insert_test_summary
from sklearn.model_selection import train_test_split

def train_and_test_pipeline(model_dir):
    """
    1) Load data from train_test_data_table
    2) 70/30 split
    3) Train model, predict test set
    4) Log row by row in test_summary_table
    5) Return summary info (section-level accuracy, chapter-level accuracy)
    6) Save model/vectorizer in model_dir
    """
    data_rows = fetch_training_data()
    if not data_rows:
        return {"error": "No data in train_test_data_table."}

    # Shuffle for randomness
    random.shuffle(data_rows)

    # Convert to X/y
    X = [d["content"] for d in data_rows]
    y = [f"{d['bookId']}-{d['chapterId']}-{d['sectionId']}" for d in data_rows]
    problem_ids = [d["problemId"] for d in data_rows]
    actual_b = [d["bookId"] for d in data_rows]
    actual_c = [d["chapterId"] for d in data_rows]
    actual_s = [d["sectionId"] for d in data_rows]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X).toarray()

    # 70/30 split
    train_size = int(len(X_vec) * 0.7)
    X_train = X_vec[:train_size]
    y_train = y[:train_size]
    X_test = X_vec[train_size:]
    y_test = y[train_size:]

    prob_test = problem_ids[train_size:]
    book_test = actual_b[train_size:]
    chap_test = actual_c[train_size:]
    sect_test = actual_s[train_size:]

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Clear old summary
    clear_test_summary()

    total = len(X_test)
    correct_section = 0
    correct_chapter = 0

    for i, vec in enumerate(X_test):
        pred_label = clf.predict([vec])[0]  # e.g. '1-24-185'
        pred_parts = pred_label.split("-")
        if len(pred_parts) == 3:
            pb, pc, ps = pred_parts
        else:
            pb, pc, ps = ("Unknown", "Unknown", "Unknown")

        ab = book_test[i]
        ac = chap_test[i]
        as_ = sect_test[i]
        pid = prob_test[i]

        match_section = 1 if (ab == pb and ac == pc and as_ == ps) else 0
        match_chapter = 1 if (ab == pb and ac == pc) else 0

        if match_section:
            correct_section += 1
        if match_chapter:
            correct_chapter += 1

        insert_test_summary(
            problem_id=pid,
            actual_b=ab, actual_c=ac, actual_s=as_,
            pred_b=pb, pred_c=pc, pred_s=ps,
            matched_section=match_section,
            matched_chapter=match_chapter
        )

    section_acc = round(correct_section / total, 3) if total else 0
    chapter_acc = round(correct_chapter / total, 3) if total else 0

    summary = {
        "test_size": total,
        "correct_section": correct_section,
        "correct_chapter": correct_chapter,
        "section_accuracy": section_acc,
        "chapter_accuracy": chapter_acc
    }

    # Save model & vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    return summary
