from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import os
import json
import numpy as np
import tensorflow as tf

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
    global local_model
    if local_model is None:
        try:
            model_path = os.path.join(MODEL_DIR, "stage1_best.keras")
            print(f"Loading local model from {model_path}...")
            local_model = tf.keras.models.load_model(model_path)
            print("Local model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    return local_model

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
        try:
            # Inference Block
            img_tensor = preprocess_image(image_bytes)
            preds = model.predict(img_tensor, verbose=0)[0]
            
            top_indices = np.argsort(preds)[::-1]
            top1_idx = top_indices[0]
            top2_idx = top_indices[1]
            
            top1_prob = float(preds[top1_idx])
            top2_prob = float(preds[top2_idx])
            top1_class = CLASS_NAMES[top1_idx]
            
            # --- QUICK DEBUG ---
            print("\n" + "="*40)
            print(f"Model confidence (Top 1): {top1_prob:.4f} ({top1_class})")
            print(f"Model Top 2 Difference : {(top1_prob - top2_prob):.4f}")
            
            # --- FINAL PRO SOLUTION LOGIC ---
            if top1_prob > 0.85 and (top1_prob - top2_prob) > 0.3:
                print("Decision: Using Local Model!")
                print("Gemini called? False")
                prediction_result = {
                    "rank": 1,
                    "confidence": top1_prob,
                    "is_uncertain": False,
                    "verified_by_gemini": False,
                    "species": {
                        "common_name": top1_class.capitalize(),
                        "scientific_name": "Unknown",
                        "family": "Unknown"
                    },
                    "details": {
                        "distinguishing_features": ["(Local inference - data limited)"],
                        "native_region": "Unknown",
                        "bloom_season": "Unknown",
                        "reference_image_url": ""
                    }
                }
            else:
                print("Decision: Confidence/Gap too low. Falling back to Gemini.")
                print("Gemini called? True")
                gemini_called = True
        except Exception as e:
            print(f"Local model error: {e}")
            gemini_called = True
    else:
        gemini_called = True
        print("Gemini called? True (Local model unavailable)")
        
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
