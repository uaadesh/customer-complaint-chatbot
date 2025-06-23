from fastapi import FastAPI, HTTPException
from .models import ComplaintCreate
from .crud import create_complaint, get_complaint_by_id
from .database import init_db

app = FastAPI(title="Customer Complaint API", version="1.0")
init_db()

@app.post("/complaints", summary="Create a new complaint", response_description="Complaint created successfully")
def create(complaint: ComplaintCreate):
    """
    Create a new complaint with name, phone number, email, and complaint details.

    Returns:
        JSON containing complaint ID and success message.
    """
    try:
        return create_complaint(complaint)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create complaint")

@app.get("/complaints/{complaint_id}", summary="Get complaint by ID", response_description="Complaint details")
def get_complaint(complaint_id: str):
    """
    Retrieve a complaint by its unique complaint ID.

    Returns:
        JSON object with complaint details if found.
    """
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint
