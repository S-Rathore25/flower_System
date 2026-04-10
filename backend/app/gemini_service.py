import os
from dotenv import load_dotenv
import logging
from PIL import Image
import io
import re
import json
import time
import base64
import requests

logger = logging.getLogger(__name__)

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}

def call_gemini_rest_with_retry(payload, retries=2):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    for i in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                print(f"DEBUG: Gemini API Error Response: {response.text}")
            response.raise_for_status()
            data = response.json()
            
            if "candidates" in data and len(data["candidates"]) > 0:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
            else:
                raise Exception("No content in Gemini response")
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(1)

# Load environment variables
load_dotenv()

# Configure API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
def ask_gemini_vision(image_bytes: bytes) -> dict:
    """
    Fallback method when local model confidence is too low.
    Uses Gemini Pro Vision to identify the flower.
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key is not set. Returning a mocked fallback.")
        return {
            "rank": 1,
            "confidence": 0.99,
            "is_uncertain": False,
            "species": {
                "common_name": "Unknown Species (Gemini Offline)",
                "scientific_name": "N/A",
                "family": "N/A"
            },
            "details": {
                "distinguishing_features": ["Need API key to identify"],
                "native_region": "Unknown",
                "bloom_season": "Unknown",
                "reference_image_url": ""
            }
        }
        
    try:
        # Load image into PIL for Gemini
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Invalid image file")
        
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt = """
You are a professional botanist.

Identify the flower in the image.

Return ONLY valid JSON. No explanation, no markdown.

Schema:
{
  "common_name": "string",
  "scientific_name": "string",
  "family": "string",
  "distinguishing_features": ["string"],
  "native_region": "string",
  "bloom_season": "string"
}

If unsure, still return best guess but keep fields filled.
"""
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        text_response = call_gemini_rest_with_retry(payload)
        
        gemini_data = extract_json(text_response)

        if not gemini_data:
            logger.warning("Gemini returned invalid JSON")
        
        return {
            "rank": 1,
            "confidence": 0.85 if gemini_data else 0.5,
            "is_uncertain": False,
            "species": {
                "common_name": gemini_data.get("common_name", "Unknown"),
                "scientific_name": gemini_data.get("scientific_name", "Unknown"),
                "family": gemini_data.get("family", "Unknown")
            },
            "details": {
                "distinguishing_features": gemini_data.get("distinguishing_features", []),
                "native_region": gemini_data.get("native_region", "Unknown"),
                "bloom_season": gemini_data.get("bloom_season", "Unknown"),
                "reference_image_url": ""  # Could ping an image search API here if needed
            }
        }
        
    except Exception as e:
        logger.error(f"Gemini API failed: {str(e)}")
        raise e
