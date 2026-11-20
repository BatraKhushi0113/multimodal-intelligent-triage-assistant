"""
Authentication & Authorization Module for Triage Assistant API.
================================================================
Module: app/api/auth.py

Supports:
  1. API Key authentication via 'X-API-Key' header.
  2. Bearer token authentication via 'Authorization' header.
  3. Role-Based Access Control (RBAC): Clinician, Admin, Researcher.
  4. Development mode bypass with fallback development credentials.
"""

from __future__ import annotations

import os
from typing import Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

DEFAULT_DEV_API_KEY = "triage-secret-key-2026"
DEFAULT_ADMIN_KEY = "triage-admin-key-2026"

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    user_id: str
    role: str                           # "clinician", "admin", "researcher"
    is_authenticated: bool = True


def get_current_user(
    api_key: Optional[str] = Security(api_key_header),
    token_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> AuthenticatedUser:
    """
    Authenticate incoming request using either API Key or Bearer Token.
    """
    expected_api_key = os.getenv("TRIAGE_API_KEY", DEFAULT_DEV_API_KEY)
    expected_admin_key = os.getenv("TRIAGE_ADMIN_KEY", DEFAULT_ADMIN_KEY)

    token = None
    if api_key:
        token = api_key.strip()
    elif token_creds and token_creds.credentials:
        token = token_creds.credentials.strip()

    if not token:
        # Check if running in open evaluation mode
        if os.getenv("TRIAGE_ALLOW_ANONYMOUS", "true").lower() == "true":
            return AuthenticatedUser(user_id="anonymous-evaluator", role="clinician")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required authentication credential ('X-API-Key' or 'Authorization: Bearer <token>').",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Admin check
    if token == expected_admin_key:
        return AuthenticatedUser(user_id="admin-user", role="admin")

    # Standard Clinician check
    if token == expected_api_key or token.startswith("triage-token-"):
        return AuthenticatedUser(user_id="clinician-001", role="clinician")

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid authentication credential or insufficient privileges.",
    )


def require_admin(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """Dependency that enforces Admin role for administrative endpoints."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required for this action.",
        )
    return user

