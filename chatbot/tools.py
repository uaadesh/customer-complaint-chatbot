# chatbot/tools.py
"""
Tools the agent can call. They use native tool calling: LangChain turns each function's
name, typed arguments and docstring into a tool schema the model sees, and the model
returns a structured call with named arguments (not JSON inside its text).

The docstrings are written for the model: they are its only description of each tool.
"""

from langchain_core.tools import tool
import requests

API_BASE = "http://localhost:8000"


@tool
def create_complaint_tool(name: str, phone_number: str, email: str, complaint_details: str) -> str:
    """File a new customer complaint. Call this only when the customer has given all four
    values themselves; never guess or invent any of them.

    Args:
        name: The customer's full name.
        phone_number: The customer's 10-digit Indian mobile number, digits only.
        email: The customer's email address.
        complaint_details: A description of the problem, in the customer's own words.
    """
    payload = {
        "name": name.strip(),
        "phone_number": phone_number.strip(),
        "email": email.strip(),
        "complaint_details": complaint_details.strip(),
    }
    try:
        res = requests.post(f"{API_BASE}/complaints", json=payload)
        if res.status_code == 200:
            return f"✅ {res.json().get('message')} (ID: {res.json().get('complaint_id')})"
        return f"❌ API error {res.status_code}: {res.text}"
    except Exception as e:
        return f"❌ Exception in complaint creation: {str(e)}"


@tool
def get_complaint_tool(complaint_id: str) -> str:
    """Look up an existing complaint by its complaint ID and return its details.

    Args:
        complaint_id: The complaint ID the customer gave, e.g. "1DCE63BD".
    """
    try:
        res = requests.get(f"{API_BASE}/complaints/{complaint_id.strip()}")
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
