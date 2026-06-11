"""
DocuAsk — AI Document Assistant
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import init_db
from app.api import documents
from app.api import documents, chat, auth    # add ", chat" here

# and add this line after app.include_router(documents.router):


app = FastAPI(
    title="DocuAsk API",
    description="Upload PDFs and ask questions about them using AI.",
    version="1.0.0",
)

# CORS — allow your React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ducoask.vercel.app", "http://localhost:5173", "http://localhost:3000"],  # Vite + CRA defaults
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Create DB tables on startup if they don't exist."""
    init_db()


# Register routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {
        "app": "DocuAsk",
        "status": "running",
        "docs": "/docs",        # FastAPI auto-generates Swagger UI here
        "redoc": "/redoc",
    }


@app.get("/health")
def health():
    return {"status": "ok"}