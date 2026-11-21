# chatbot/tools.py

from langchain.tools import tool
import requests

API_BASE = "http://localhost:8000"

@tool
def create_complaint_tool(data: str) -> str:
    """
    Tool: Create a new complaint.
    
    Input (str): A comma-separated string of 4 values:
        "name, phone_number, email, complaint_details"
    Only the first 3 commas separate fields, so the complaint details
    may themselves contain commas.

    Example:
        "Deepak, 9876543210, deepak@example.com, Order late, box damaged"

    Returns:
        str: Success or error message
    """
    try:
        # maxsplit=3: name, phone and email never contain commas, so everything
        # after the third comma is the complaint text, commas included.
        parts = [x.strip() for x in data.split(",", 3)]
        if len(parts) != 4:
            return "Error: Please provide exactly 4 comma-separated fields: name, phone, email, details."

        payload = {
            "name": parts[0],
            "phone_number": parts[1],
            "email": parts[2],
            "complaint_details": parts[3],
        }

        res = requests.post(f"{API_BASE}/complaints", json=payload)
        if res.status_code == 200:
            return f"✅ {res.json().get('message')} (ID: {res.json().get('complaint_id')})"
        else:
            return f"❌ API error {res.status_code}: {res.text}"
    except Exception as e:
        return f"❌ Exception in complaint creation: {str(e)}"


@tool
def get_complaint_tool(complaint_id: str) -> str:
    """
    Tool: Retrieve complaint details by ID.
    
    Input:
        complaint_id (str): The unique complaint ID.
    
    Returns:
        str: Formatted complaint details or error message.
    """
    try:
        res = requests.get(f"{API_BASE}/complaints/{complaint_id}")
        if res.status_code != 200:
            return f"❌ Complaint not found or server error (status code: {res.status_code})"

        data = res.json()
        return (
            f"📄 Complaint ID: {data['complaint_id']}\n"
            f"👤 Name: {data['name']}\n"
            f"📞 Phone: {data['phone_number']}\n"
            f"📧 Email: {data['email']}\n"
            f"📝 Details: {data['complaint_details']}\n"
            f"🕒 Created At: {data['created_at']}"
        )
    except Exception as e:
        return f"❌ Exception retrieving complaint: {str(e)}"
