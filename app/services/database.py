"""
PriceIQ Pro — Database Persistence Layer v1.2
With detailed error logging for debugging.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import logging

from app.core.config import settings

# Optional Supabase import
try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError as e:
    _SUPABASE_AVAILABLE = False
    _SUPABASE_ERROR = str(e)

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self._client: Optional[Any] = None
        self._available = False
        
        # Log what we have
        logger.info("=" * 50)
        logger.info("Supabase Connection Debug:")
        logger.info(f"  supabase package available: {_SUPABASE_AVAILABLE}")
        if not _SUPABASE_AVAILABLE:
            logger.error(f"  Import error: {_SUPABASE_ERROR}")
        logger.info(f"  SUPABASE_URL: {'SET' if settings.SUPABASE_URL else 'NOT SET'}")
        if settings.SUPABASE_URL:
            logger.info(f"  URL: {settings.SUPABASE_URL[:50]}...")
        logger.info(f"  SUPABASE_KEY: {'SET' if settings.SUPABASE_KEY else 'NOT SET'}")
        if settings.SUPABASE_KEY:
            logger.info(f"  KEY (first 20 chars): {settings.SUPABASE_KEY[:20]}...")
        logger.info("=" * 50)

        if _SUPABASE_AVAILABLE and settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                logger.info("Attempting to create Supabase client...")
                self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                logger.info("Supabase client created successfully")
                
                # Test the connection with a simple query
                logger.info("Testing connection with a simple query...")
                try:
                    result = self._client.table("users").select("count", count="exact").limit(0).execute()
                    logger.info(f"✅ Supabase connected successfully! Users table accessible.")
                    self._available = True
                except Exception as e:
                    logger.error(f"❌ Supabase table query failed: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    if hasattr(e, 'message'):
                        logger.error(f"Message: {e.message}")
                    self._available = False
                    
            except Exception as e:
                logger.error(f"❌ Supabase client creation failed: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                if hasattr(e, 'message'):
                    logger.error(f"Message: {e.message}")
                self._available = False
        else:
            missing = []
            if not _SUPABASE_AVAILABLE:
                missing.append("supabase package")
            if not settings.SUPABASE_URL:
                missing.append("SUPABASE_URL")
            if not settings.SUPABASE_KEY:
                missing.append("SUPABASE_KEY")
            logger.warning(f"❌ Supabase not configured - missing: {', '.join(missing)}")

    @property
    def is_available(self) -> bool:
        return self._available

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            result = self._client.table("users").select("*").eq("id", user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"get_user_by_id error: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            result = self._client.table("users").select("*").eq("email", email.lower().strip()).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"get_user_by_email error: {e}")
            return None

    async def validate_api_key(self, raw_key: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            hashed = hashlib.sha256(raw_key.encode()).hexdigest()
            result = self._client.table("api_keys").select("*, users(*)").eq("key_hash", hashed).eq("is_active", True).execute()
            if result.data:
                return result.data[0].get("users")
            return None
        except Exception as e:
            logger.error(f"validate_api_key error: {e}")
            return None

    async def get_signals(self, user_id: Optional[str] = None, pair: Optional[str] = None, limit: int = 50) -> List[Dict]:
        if not self._available:
            return []
        try:
            query = self._client.table("signals").select("*").order("created_at", desc=True).limit(limit)
            if user_id:
                query = query.eq("user_id", user_id)
            if pair:
                query = query.eq("pair", pair.upper())
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"get_signals error: {e}")
            return []


# Singleton
db = Database()
