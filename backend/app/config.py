import os

class Settings:
    PROJECT_NAME: str = "Flower Species Identification System"
    API_V1_STR: str = "/api/v1"
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/flower_model.h5")
    CONFIDENCE_THRESHOLD: float = 0.60
    
    # Database setttings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./flowers.db")

settings = Settings()
