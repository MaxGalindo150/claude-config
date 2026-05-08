---
name: fastapi-security
description: FastAPI security best practices — JWT authentication with python-jose, OAuth2, role-based access, CORS, rate limiting, security headers, and input validation.
origin: local
---

# FastAPI Security Best Practices

Comprehensive security patterns for FastAPI APIs using python-jose, OAuth2, and production hardening.

## When to Activate

- Setting up authentication and authorization in FastAPI
- Implementing JWT-based auth with python-jose
- Configuring CORS, rate limiting, security headers
- Reviewing FastAPI apps for security issues
- Deploying FastAPI to production

## JWT Authentication with python-jose

### Token Utilities

```python
# app/auth/tokens.py
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from app.config import settings


def create_access_token(subject: str | int, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": str(subject), "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> str:
    """Decode and validate JWT. Returns subject or raises JWTError."""
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    sub: str | None = payload.get("sub")
    if sub is None:
        raise JWTError("Missing subject")
    return sub
```

### Password Hashing

```python
# app/auth/passwords.py
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

### OAuth2 Dependencies

```python
# app/dependencies.py
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tokens import decode_access_token
from app.models.user import User
from app.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_access_token(token)
    except JWTError:
        raise credentials_exception

    user = await db.get(User, int(user_id))
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user


# Reusable annotated types
CurrentUser = Annotated[User, Depends(get_current_active_user)]
```

### Auth Router

```python
# app/routers/auth.py
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.passwords import verify_password
from app.auth.tokens import create_access_token
from app.dependencies import DbSession
from app.models.user import User
from app.schemas.auth import Token
from sqlalchemy import select

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    access_token = create_access_token(subject=user.id)
    return Token(access_token=access_token, token_type="bearer")


# app/schemas/auth.py
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str
```

## Role-Based Access Control (RBAC)

```python
# app/auth/roles.py
from enum import StrEnum
from fastapi import HTTPException, status
from app.models.user import User


class Role(StrEnum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


def require_role(*roles: Role):
    """Dependency factory that requires one of the given roles."""
    async def check_role(current_user: CurrentUser) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user
    return check_role


# Usage
AdminOnly = Annotated[User, Depends(require_role(Role.ADMIN))]
ModeratorOrAbove = Annotated[User, Depends(require_role(Role.MODERATOR, Role.ADMIN))]


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: int, _: AdminOnly, db: DbSession):
    ...
```

## CORS Configuration

```python
# app/main.py
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # Never use ["*"] in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# config.py — parse from env
cors_origins: list[str] = ["https://app.example.com"]
```

## Rate Limiting with slowapi

```python
# app/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# app/routers/auth.py
from app.main import limiter
from fastapi import Request

@router.post("/token")
@limiter.limit("10/minute")
async def login(request: Request, ...):
    ...

# Stricter limit for sensitive endpoints
@router.post("/register")
@limiter.limit("5/hour")
async def register(request: Request, ...):
    ...
```

## Security Headers Middleware

```python
# app/middleware/security_headers.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        return response


# app/main.py
app.add_middleware(SecurityHeadersMiddleware)
```

## Input Validation

```python
# Pydantic v2 — use Field constraints, never trust user input
from pydantic import BaseModel, Field, EmailStr, field_validator
import re

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v
```

## File Upload Security

```python
import magic  # python-magic
from fastapi import UploadFile, HTTPException

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "application/pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


async def validate_upload(file: UploadFile) -> bytes:
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    # Validate actual content, not just the declared content-type
    detected = magic.from_buffer(content, mime=True)
    if detected not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {detected}")

    return content
```

## Secrets Management

```python
# app/config.py — all secrets from environment, never hardcoded
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # SecretStr prevents accidental logging
    secret_key: SecretStr
    database_url: SecretStr
    redis_url: SecretStr | None = None

    def get_secret_key(self) -> str:
        return self.secret_key.get_secret_value()
```

## Logging Security Events

```python
# app/middleware/audit_log.py
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

security_logger = logging.getLogger("security")


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Log auth failures and suspicious activity
        if response.status_code in (401, 403):
            security_logger.warning(
                "Auth failure",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ip": request.client.host if request.client else "unknown",
                    "duration_ms": round(duration * 1000, 2),
                },
            )
        return response
```

## Security Checklist

| Check | How |
|-------|-----|
| Never expose `DEBUG=True` in prod | `settings.debug = False` |
| JWT secret via env var | `SecretStr` in settings |
| Password hashing | `bcrypt` via `passlib` |
| HTTPS only | Enforce at reverse proxy (nginx/Caddy) |
| CORS locked down | Explicit origins, not `["*"]` |
| Rate limit auth endpoints | `slowapi` on `/token`, `/register` |
| Validate file uploads | Content-type via `python-magic`, size limit |
| Security headers | `SecurityHeadersMiddleware` |
| Input validation | Pydantic `Field` constraints + validators |
| No SQL injection | SQLAlchemy ORM or parameterized queries only |
| Log auth failures | `AuditLogMiddleware` |
| Dependency updates | `pip-audit` in CI |

Remember: No CSRF protection needed for stateless JWT APIs (no cookies carrying session), but always verify `Origin` for cookie-based auth.
