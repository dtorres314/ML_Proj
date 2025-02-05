import random, os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.db_manager import (
    fetch_training_data_for_book,
    clear_test_summary,
    insert_test_summary
)

def train_and_test_pipeline(model_dir):
    """
    1) Only fetch data from Book 1
    2) Shuffle, 70/30 split
    3) Train model, test -> log results
    4) Save model & vectorizer to model_dir
    5) Return summary stats
    """
    # *** Only load Book 1's data ***
    data_rows = fetch_training_data_for_book("1")
    if not data_rows:
        return {"error": "No data found for Book 1 in DB."}

    random.shuffle(data_rows)

    X = [d["content"] for d in data_rows]
    y = [f"{d['bookId']}-{d['chapterId']}-{d['sectionId']}" for d in data_rows]
    prob_ids = [d["problemId"] for d in data_rows]
    act_books = [d["bookId"] for d in data_rows]
    act_chaps = [d["chapterId"] for d in data_rows]
    act_secs = [d["sectionId"] for d in data_rows]

    vec = TfidfVectorizer(max_features=5000)
    X_mat = vec.fit_transform(X).toarray()

    train_size = int(len(X_mat)*0.7)
    X_train, y_train = X_mat[:train_size], y[:train_size]
    X_test, y_test = X_mat[train_size:], y[train_size:]

    test_prob_ids = prob_ids[train_size:]
    test_books = act_books[train_size:]
    test_chaps = act_chaps[train_size:]
    test_secs = act_secs[train_size:]

    if not X_test:
        return {"error": "Not enough data to create a test split. Need >1 problem for train & test."}

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Clear old summary
    clear_test_summary()

    total = len(X_test)
    correct_section = 0
    correct_chapter = 0

    for i, xv in enumerate(X_test):
        pred_label = clf.predict([xv])[0]  # e.g. '1-2-185'
        p_parts = pred_label.split("-")
        if len(p_parts) == 3:
            pb, pc, ps = p_parts
        else:
            pb, pc, ps = ("Unknown", "Unknown", "Unknown")

        ab = test_books[i]
        ac = test_chaps[i]
        as_ = test_secs[i]
        pid = test_prob_ids[i]

        match_section = 1 if (pb==ab and pc==ac and ps==as_) else 0
        match_chapter = 1 if (pb==ab and pc==ac) else 0

        if match_section:
            correct_section += 1
        if match_chapter:
            correct_chapter += 1

        insert_test_summary(
            problem_id=pid,
            actual_b=ab,
            actual_c=ac,
            actual_s=as_,
            pred_b=pb,
            pred_c=pc,
            pred_s=ps,
            matched_section=match_section,
            matched_chapter=match_chapter
        )

    section_acc = round(correct_section / total, 3)
    chapter_acc = round(correct_chapter / total, 3)
    summary = {
        "test_size": total,
        "section_correct": correct_section,
        "chapter_correct": correct_chapter,
        "section_accuracy": section_acc,
        "chapter_accuracy": chapter_acc
    }

    # Save model & vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vec, os.path.join(model_dir, "vectorizer.pkl"))

    return summary
