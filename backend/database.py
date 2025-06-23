import sqlite3
import logging

# Path to the SQLite database file
DB_PATH = "complaints.db"

def get_db():
    """
    Returns a connection to the SQLite database.
    Caller must ensure the connection is closed after use.
    """
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def init_db():
    """
    Initializes the complaints table in the database.
    If the table already exists, this has no effect.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    complaint_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    email TEXT NOT NULL,
                    complaint_details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error initializing database: {e}")
        raise
