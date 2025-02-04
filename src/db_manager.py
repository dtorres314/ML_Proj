import sqlite3

DB_NAME = "mydatabase.db"

def init_db():
    """
    Ensures the DB and main tables exist.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Main data table
    c.execute("""
        CREATE TABLE IF NOT EXISTS train_test_data_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problemId TEXT,
            bookId TEXT,
            chapterId TEXT,
            sectionId TEXT,
            problemContent TEXT
        )
    """)

    # Summary table for test predictions
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_summary_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problemId TEXT,
            actualBook TEXT,
            actualChapter TEXT,
            actualSection TEXT,
            predBook TEXT,
            predChapter TEXT,
            predSection TEXT,
            matchedSection INTEGER,
            matchedChapter INTEGER
        )
    """)

    conn.commit()
    conn.close()

def insert_problem_entry(problem_id, book_id, chapter_id, section_id, content):
    """
    Insert a single row into train_test_data_table.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO train_test_data_table
        (problemId, bookId, chapterId, sectionId, problemContent)
        VALUES (?, ?, ?, ?, ?)
    """, (problem_id, book_id, chapter_id, section_id, content))
    conn.commit()
    conn.close()

def clear_test_summary():
    """
    Clears the test_summary_table before each new train/test run.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM test_summary_table")
    conn.commit()
    conn.close()

def insert_test_summary(problem_id, actual_b, actual_c, actual_s,
                        pred_b, pred_c, pred_s,
                        matched_section, matched_chapter):
    """
    Insert a single row into test_summary_table with the test outcome.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO test_summary_table
        (problemId, actualBook, actualChapter, actualSection,
         predBook, predChapter, predSection,
         matchedSection, matchedChapter)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (problem_id, actual_b, actual_c, actual_s,
          pred_b, pred_c, pred_s,
          matched_section, matched_chapter))
    conn.commit()
    conn.close()

def fetch_training_data():
    """
    Loads all rows from train_test_data_table as a list of dicts.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM train_test_data_table")
    rows = c.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "problemId": r["problemId"],
            "bookId": r["bookId"],
            "chapterId": r["chapterId"],
            "sectionId": r["sectionId"],
            "content": r["problemContent"]
        })
    return data
