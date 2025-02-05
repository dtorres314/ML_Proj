import os
import joblib
import random
import numpy as np

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
    1) Fetch data for Book 1 from the DB (train_test_data_table).
    2) Use sklearn's train_test_split with test_size=0.3 => 70% train, 30% test.
    3) Train RandomForest on the training set, then predict on the test set.
    4) For each tested example, log results in test_summary_table (actual vs. predicted).
    5) Return summary stats for matched chapter and matched section, and save model to model_dir.
    """
    # 1) Load data for Book 1 only
    data_rows = fetch_training_data_for_book("1")
    if not data_rows:
        return {"error": "No data found for Book 1 in DB."}

    # 2) Prepare data
    X_raw = [row["content"] for row in data_rows]
    y_raw = [f"{row['bookId']}-{row['chapterId']}-{row['sectionId']}" for row in data_rows]

    problem_ids = [row["problemId"] for row in data_rows]
    books = [row["bookId"] for row in data_rows]
    chapters = [row["chapterId"] for row in data_rows]
    sections = [row["sectionId"] for row in data_rows]

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_matrix = vectorizer.fit_transform(X_raw).toarray()

    # 3) Use train_test_split with test_size=0.3 => 70% for training, 30% for testing
    (X_train, X_test,
     y_train, y_test,
     pid_train, pid_test,
     book_train, book_test,
     chap_train, chap_test,
     sec_train, sec_test) = train_test_split(
         X_matrix, y_raw,
         problem_ids, books, chapters, sections,
         test_size=0.3,
         random_state=42,
         shuffle=True
    )

    # If no test data
    if len(X_test) == 0:
        return {"error": "Not enough data to form a test set. Possibly only one problem in DB."}

    # 4) Train a RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Clear old summary
    clear_test_summary()

    total = len(X_test)
    correct_section = 0
    correct_chapter = 0

    # 5) Predict each test sample
    for i, xvec in enumerate(X_test):
        pred_label = clf.predict([xvec])[0]  # e.g. "1-24-185"
        parts = pred_label.split("-")
        if len(parts) == 3:
            pb, pc, ps = parts
        else:
            pb, pc, ps = ("Unknown", "Unknown", "Unknown")

        actual_book = book_test[i]
        actual_chap = chap_test[i]
        actual_sec = sec_test[i]
        prob_id = pid_test[i]

        match_section = 1 if (pb == actual_book and pc == actual_chap and ps == actual_sec) else 0
        match_chapter = 1 if (pb == actual_book and pc == actual_chap) else 0

        if match_section:
            correct_section += 1
        if match_chapter:
            correct_chapter += 1

        # Log in test_summary_table
        insert_test_summary(
            problem_id=prob_id,
            actual_b=actual_book,
            actual_c=actual_chap,
            actual_s=actual_sec,
            pred_b=pb,
            pred_c=pc,
            pred_s=ps,
            matched_section=match_section,
            matched_chapter=match_chapter
        )

    # Summaries
    section_acc = round(correct_section / total, 3)
    chapter_acc = round(correct_chapter / total, 3)

    summary = {
        "test_size": total,
        "section_correct": correct_section,
        "chapter_correct": correct_chapter,
        "section_accuracy": section_acc,
        "chapter_accuracy": chapter_acc
    }

    # 6) Save trained model & vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    return summary
