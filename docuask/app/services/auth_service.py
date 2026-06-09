"""
auth_service.py — handles everything authentication related:
  - password hashing with bcrypt
  - JWT token creation and verification
  - user registration and login
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import User

settings = get_settings()

# bcrypt context — handles password hashing and verification
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)

# OAuth2 scheme — looks for Bearer token in Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Convert plain password to bcrypt hash. Truncate to 72 chars (bcrypt limit)."""
    return pwd_context.hash(password[:72])


def verify_password(plain: str, hashed: str) -> bool:
    """Check if a plain password matches its hash."""
    return pwd_context.verify(plain[:72], hashed)

# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT token.
    The token contains the user's email (sub) and expiry time.
    It is signed with SECRET_KEY so it cannot be tampered with.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")


def decode_token(token: str) -> Optional[str]:
    """
    Decode and verify a JWT token.
    Returns the email (sub) if valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload.get("sub")
    except JWTError:
        return None


# ── User CRUD ─────────────────────────────────────────────────────────────────

def get_user_by_email(email: str, db: Session) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(username: str, db: Session) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def create_user(email: str, username: str, password: str, db: Session) -> User:
    """Register a new user. Raises if email or username already taken."""
    if get_user_by_email(email, db):
        raise HTTPException(status_code=400, detail="Email already registered.")
    if get_user_by_username(username, db):
        raise HTTPException(status_code=400, detail="Username already taken.")

    user = User(
        email=email,
        username=username,
        hashed_password=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(email: str, password: str, db: Session) -> Optional[User]:
    """Verify email + password. Returns user if correct, None if not."""
    user = get_user_by_email(email, db)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# ── FastAPI dependency — get current logged-in user ───────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency injected into protected endpoints.
    Reads Bearer token from request header, decodes it,
    and returns the User object.

    Usage in any endpoint:
      def my_endpoint(current_user: User = Depends(get_current_user)):
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = decode_token(token)
    if not email:
        raise credentials_exception

    user = get_user_by_email(email, db)
    if not user or not user.is_active:
        raise credentials_exception

    return user