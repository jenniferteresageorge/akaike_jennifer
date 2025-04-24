from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from models import EmailClassifier
from utils import PIIDetector
import joblib
import os

app = FastAPI(
    title="Email Classification and PII Masking API",
    description="API for classifying support emails and masking PII information",
    version="1.0.0"
)

# Initialize components
pii_detector = PIIDetector()
email_classifier = EmailClassifier()

try:
    email_classifier.load_model("email_classifier.joblib")
except Exception as e:
    print("Model loading failed:", e)
    raise RuntimeError("Pre-trained model not found. Please train it using train_model.py")


class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
        position: List[int]
        classification: str
        entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

@app.post("/classify_email", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """
    Endpoint for classifying emails and masking PII.
    
    Args:
        request: EmailRequest containing the email body
        
    Returns:
        EmailResponse with classification and PII masking information
    """
    try:
        # Step 1: Detect PII in the email
        email_text = request.email_body
        detected_entities = pii_detector.detect_pii(email_text)
        
        # Step 2: Mask the PII
        masked_email, masked_entities = pii_detector.mask_pii(email_text, detected_entities)
        
        # Step 3: Classify the email
        category = email_classifier.predict(masked_email)
        
        # Prepare response
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
