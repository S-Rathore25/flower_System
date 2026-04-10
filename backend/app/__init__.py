import sys
import traceback

print("Starting to load backend/app/__init__.py...")
try:
    from .main import app
    print("Successfully imported app from .main!")
except Exception as e:
    print("FATAL IMPORT ERROR:", e)
    traceback.print_exc()
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/")
    def index(): return {"error": str(e)}
