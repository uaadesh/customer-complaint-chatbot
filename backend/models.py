from pydantic import BaseModel, EmailStr, field_validator, Field
import re

class ComplaintCreate(BaseModel):
    """
    Pydantic model to validate incoming complaint data.
    Ensures proper formatting of name, phone number, email, and complaint details.
    """
    name: str = Field(..., example="Deepak", description="Customer's full name")
    phone_number: str = Field(..., example="9876543210", description="10-digit Indian phone number starting with 6-9")
    email: EmailStr = Field(..., example="deepak@example.com", description="Valid email address of the customer")
    complaint_details: str = Field(..., example="Product not delivered on time", description="Details of the complaint")

    @field_validator("phone_number")
    def validate_indian_phone_number(cls, v: str) -> str:
        """
        Validates that the phone number:
        - Starts with 6, 7, 8, or 9
        - Is exactly 10 digits long
        """
        if not re.fullmatch(r'[6-9]\d{9}', v):
            raise ValueError("Phone number must be 10 digits starting with 6-9.")
        return v.strip()
