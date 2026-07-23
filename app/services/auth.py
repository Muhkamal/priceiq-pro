"""
PriceIQ Pro — Authentication & Authorization v3.2
"""

import hashlib
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Set

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

logger = logging.getLogger(__name__)

# -- Hard require PyJWT --
try:
    import jwt
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
except ImportError:
    raise ImportError(
        "PyJWT is required. Install it: pip install PyJWT"
    )

# -- Hard require bcrypt --
try:
    import bcrypt
except ImportError:
    raise ImportError(
        "bcrypt is required. Install it: pip install bcrypt"
    )

# Lazy imports
_settings = None
_db = None

def _get_settings():
    global _settings
    if _settings is None:
        from app.core.config import settings as s
        _settings = s
    return _settings

def _get_db():
    global _db
    if _db is None:
        from app.services.database import db as d
        _db = d
    return _db

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
JWT_ALGORITHM = "HS256"

_auth_validated = False

def validate_auth_config():
    global _auth_validated
    s = _get_settings()
    if not s.JWT_SECRET or s.JWT_SECRET in ("change_this_in_production", ""):
        raise RuntimeError(
            "JWT_SECRET is not configured. Generate one with:\n"
            "python3 -c \"import secrets; print(secrets.token_hex(32))\""
        )
    _auth_validated = True
    logger.info("Auth config validated")

def hash_password(password: str, rounds: int = 12) -> str:
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def _jwt_secret() -> str:
    return _get_settings().JWT_SECRET

def create_access_token(user_id: str, email: str, tier: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "tier": tier,
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str):
    jti = secrets.token_hex(16)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "jti": jti,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM), jti

def decode_token(token: str) -> Dict:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

_token_blacklist: Set[str] = set()

def blacklist_token(token: str):
    _token_blacklist.add(token)

def is_token_blacklisted(token: str) -> bool:
    return token in _token_blacklist

_signal_usage: Dict[str, Dict] = {}

def check_signal_rate_limit(user: Dict) -> bool:
    user_id = user.get("id", "anonymous")
    tier = user.get("tier", "free")
    limit = _get_tier_limit(tier)
    today = datetime.now(timezone.utc).date().isoformat()
    if user_id not in _signal_usage:
        _signal_usage[user_id] = {"count": 0, "date": today}
    usage = _signal_usage[user_id]
    if usage["date"] != today:
        usage["count"] = 0
        usage["date"] = today
    return usage["count"] < limit

def increment_signal_usage(user_id: str):
    today = datetime.now(timezone.utc).date().isoformat()
    if user_id not in _signal_usage:
        _signal_usage[user_id] = {"count": 0, "date": today}
    if _signal_usage[user_id]["date"] != today:
        _signal_usage[user_id] = {"count": 0, "date": today}
    _signal_usage[user_id]["count"] += 1

def get_remaining_signals(user: Dict) -> int:
    user_id = user.get("id", "anonymous")
    tier = user.get("tier", "free")
    limit = _get_tier_limit(tier)
    today = datetime.now(timezone.utc).date().isoformat()
    usage = _signal_usage.get(user_id, {"count": 0, "date": today})
    if usage["date"] != today:
        return limit
    return max(0, limit - usage["count"])

def _get_tier_limit(tier: str) -> int:
    s = _get_settings()
    return {
        "free": s.FREE_DAILY_SIGNALS,
        "starter": s.STARTER_DAILY_SIGNALS,
        "pro": s.PRO_DAILY_SIGNALS,
        "elite": s.ELITE_DAILY_SIGNALS,
    }.get(tier, s.FREE_DAILY_SIGNALS)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if is_token_blacklisted(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = _get_db()
    if db.is_available:
        user = await db.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    else:
        return {
            "id": user_id,
            "email": payload.get("email"),
            "tier": payload.get("tier", "free"),
            "_db_fallback": True,
        }

async def get_user_from_api_key(api_key: Optional[str] = Security(api_key_header)) -> Dict:
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key required")
    if not api_key.startswith("priceiq_") or len(api_key) < 20:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    db = _get_db()
    user = await db.validate_api_key(api_key)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return user

async def get_current_user_or_api_key(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> Dict:
    if bearer:
        return await get_current_user(bearer)
    if api_key:
        return await get_user_from_api_key(api_key)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

async def require_signal_quota(user: Dict = Depends(get_current_user_or_api_key)) -> Dict:
    if not check_signal_rate_limit(user):
        tier = user.get("tier", "free")
        limit = _get_tier_limit(tier)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Daily limit reached ({limit})")
    return user

async def require_pro_tier(user: Dict = Depends(get_current_user_or_api_key)) -> Dict:
    if user.get("tier") not in ("pro", "elite"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Pro tier required")
    return user

async def require_elite_tier(user: Dict = Depends(get_current_user_or_api_key)) -> Dict:
    if user.get("tier") != "elite":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Elite tier required")
    return user

async def require_admin(user: Dict = Depends(get_current_user_or_api_key)) -> Dict:
    if user.get("tier") not in ("admin", "elite"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return user

async def get_optional_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[Dict]:
    try:
        return await get_current_user_or_api_key(bearer, api_key)
    except HTTPException:
        return None

async def logout_user(token: str):
    blacklist_token(token)

def run_periodic_cleanup():
    global _signal_usage
    today = datetime.now(timezone.utc).date().isoformat()
    _signal_usage = {k: v for k, v in _signal_usage.items() if v["date"] == today}
