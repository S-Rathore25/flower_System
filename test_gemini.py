import requests
import json
import base64

API_KEY = "AIzaSyCyyxprZoO5txGVzTlo1H5fLV9Wrxg5Mpc"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

# simple transparent 1x1 gif converted to jpeg for testing
img_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="

prompt = "Identify"
payload = {
    "contents": [{
        "parts": [
            {"text": prompt},
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_b64
                }
            }
        ]
    }],
    "generationConfig": {
        "responseMimeType": "application/json"
    }
}
response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
print(response.status_code)
print(response.text)
