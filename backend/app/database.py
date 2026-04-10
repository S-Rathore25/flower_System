from sqlalchemy import create_url, create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./flowers.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class IdentificationLog(Base):
    __tablename__ = "identification_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    top_pred_class = Column(String)
    top_pred_conf = Column(Float)
    all_predictions = Column(JSON)
    processing_ms = Column(Integer)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(Integer)
    is_correct = Column(Boolean)
    actual_species = Column(String)
    comment = Column(String)

Base.metadata.create_all(bind=engine)
