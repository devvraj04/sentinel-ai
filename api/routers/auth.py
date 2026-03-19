"""
api/routers/auth.py
JWT-based authentication for dashboard users.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
 
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
 
from config.settings import get_settings
 
settings = get_settings()
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
 
 
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: str
 
 
class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
 
 
def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode({**data, "exp": expire}, settings.secret_key, algorithm=settings.jwt_algorithm)
 
 
@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT token."""
    # In production: query PostgreSQL for the user
    # For dev: accept the default admin user
    if form.username == "admin@sentinel.bank" and form.password == "sentinel_admin":
        token = create_access_token({"sub": "admin-001", "role": "admin", "name": "System Admin"})
        return Token(access_token=token, token_type="bearer", role="admin", full_name="System Admin")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
