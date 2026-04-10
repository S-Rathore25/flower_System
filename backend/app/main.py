from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import os
import json
import numpy as np

from .preprocessing import preprocess_image
from .gemini_service import ask_gemini_vision

app = FastAPI(title="Flower Identification API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the local model lazily
MODEL_DIR = r"c:\Users\ABC\Desktop\anigrevity\Flower_System\models"
try:
    with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"Loaded {len(CLASS_NAMES)} classes.")
except Exception:
    CLASS_NAMES = []
    
local_model = None

def get_model():
    return None

def predict(image):
    return "sunflower"

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/v1/identify")
def identify_flower(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE_FORMAT")
        
    start_time = time.time()
    image_bytes = file.file.read() # Note: changed from await file.read() since it's now sync
    
    request_id = str(uuid.uuid4())
    
    model = get_model()
    gemini_called = False
    prediction_result = None
    
    if model and CLASS_NAMES:
        pass # Not using local model anymore
    else:
        # User requested dummy model to test deployment
        dummy_result = predict(image_bytes)
        print(f"Dummy model predicted: {dummy_result}")
        gemini_called = True
        print("Gemini called? True (Local model replaced with dummy)")
        
    if gemini_called:
        print("Calling Gemini API...")
        try:
            prediction_result = ask_gemini_vision(image_bytes)
            prediction_result["verified_by_gemini"] = True
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Identification failed. Gemini API error: {str(e)}. Please check your API key."
            )
        
    print("="*40 + "\n")
        
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "request_id": request_id,
        "timestamp": time.time(),
        "processing_time_ms": processing_time_ms,
        "predictions": [prediction_result],
        "metadata": {
            "model_version": "v3.0-hybrid",
            "gemini_fallback_used": gemini_called
        }
    }
