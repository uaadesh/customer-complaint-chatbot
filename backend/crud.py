import uuid
from datetime import datetime
from sqlite3 import Error
from .database import get_db
import logging

def create_complaint(data) -> dict:
    """
    Inserts a new complaint into the database.

    Args:
        data: An instance of ComplaintCreate with user input.

    Returns:
        dict: Contains generated complaint_id and a success message.
    """
    complaint_id = str(uuid.uuid4())[:8].upper()  # Short, unique ID
    now = datetime.now().isoformat()              # ISO 8601 timestamp

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    complaint_id,
                    data.name.strip(),
                    data.phone_number.strip(),
                    data.email.strip(),
                    data.complaint_details.strip(),
                    now
                )
            )
            conn.commit()
    except Error as e:
        logging.error(f"Database insert error: {e}")
        raise

    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully"
    }


def get_complaint_by_id(complaint_id: str) -> dict | None:
    """
    Retrieves a complaint from the database using its ID.

    Args:
        complaint_id (str): The ID of the complaint to retrieve.

    Returns:
        dict | None: Complaint details if found, else None.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
            row = cursor.fetchone()
    except Error as e:
        logging.error(f"Database read error: {e}")
        raise

    if row:
        return {
            "complaint_id": row[0],
            "name": row[1],
            "phone_number": row[2],
            "email": row[3],
            "complaint_details": row[4],
            "created_at": row[5]
        }

    return None
