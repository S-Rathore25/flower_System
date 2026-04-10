import sys
import traceback

print("Starting to load app.py...")
try:
    from backend.app.main import app
    print("Successfully imported app from backend.app.main!")
except Exception as e:
    print("FATAL IMPORT ERROR:", e)
    traceback.print_exc()
    # Provide a dummy app so it doesn't crash on import, but we can read the logs!
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/")
    def index(): return {"error": str(e)}
