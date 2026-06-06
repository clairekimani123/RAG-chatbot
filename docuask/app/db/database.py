from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Run once on startup:
    1. Enable the pgvector extension in PostgreSQL
    2. Create all tables defined in models
    """
    with engine.connect() as conn:
        # Enable pgvector — must be installed on your Postgres server
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Import models here so Base knows about them before create_all
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")