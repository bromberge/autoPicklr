# db.py
import os
from sqlmodel import Session, create_engine

# Point this at the SAME database your app already uses.
# If you don't have a DATABASE_URL yet, this sqlite file will be created.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")

engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    """FastAPI dependency: yields a SQLModel Session."""
    with Session(engine) as session:
        yield session
