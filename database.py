
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./pcos_compass.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class User(Base):
    """Store each user's account information."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    symptoms = relationship(
        "SymptomEntryDB",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class SymptomEntryDB(Base):
    """Store symptom tracker entries for an individual user."""

    __tablename__ = "symptom_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    period_start = Column(String, default="")
    period_end = Column(String, default="")
    pain = Column(Integer, default=0)
    acne = Column(String, default="none")
    mood = Column(String, default="neutral")
    energy = Column(Integer, default=0)
    weight = Column(Float, nullable=True)
    sleep = Column(Float, default=0)
    medication = Column(String, default="")
    notes = Column(String, default="")

    date = Column(String, nullable=False)
    period_duration = Column(Integer, default=1)
    suggestions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="symptoms")


Base.metadata.create_all(bind=engine)