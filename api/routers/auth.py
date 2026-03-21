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


def _get_db_connection():
    """Get a synchronous psycopg2 connection for user lookup."""
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(
        settings.database_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user against PostgreSQL users table with bcrypt verification."""
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, full_name, role, password_hash FROM users WHERE email = %s AND is_active = TRUE",
                (form.username,),
            )
            user = cur.fetchone()
        conn.close()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    if not user or not pwd_context.verify(form.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({
        "sub": str(user["id"]),
        "role": user["role"],
        "name": user["full_name"],
    })
    return Token(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        full_name=user["full_name"],
    )

