# Flower Identification System

An AI-powered web application for identifying flower species using Deep Learning and Gemini fallback.

## Project Structure
- `backend/`: FastAPI server for processing images and model inference.
- `frontend/`: React + Vite + Tailwind CSS dashboard.
- `ml_model/`: Python scripts for data collection and model training.
- `models/`: Trained model files and class configurations.

## Setup

### Backend
1. Navigate to `backend`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Start the server: `python main.py` or `uvicorn app.main:app`.

### Frontend
1. Navigate to `frontend`.
2. Install dependencies: `npm install`.
3. Start development server: `npm run dev`.

## Technologies
- **Frontend**: React, Tailwind CSS, Vite
- **Backend**: FastAPI, TensorFlow/Keras
- **AI**: Custom CNN, Google Gemini API fallback
