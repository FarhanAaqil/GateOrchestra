"""
api/account_routes.py
=====================
A signed-in person's own settings in hosted mode (404 outside it).

  GET    /account/keys              which providers have a key of yours (never the key)
  PUT    /account/keys              store your own Groq or Gemini key (checked first)
  DELETE /account/keys/{provider}   forget it
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth_routes import chat_owner
from assistant.service import KeyRejected, get_service
from assistant.userkeys import KeyStoreError

router = APIRouter()


class OwnKey(BaseModel):
    provider: str = Field(..., pattern="^(groq|gemini)$")
    key: str = Field(..., min_length=20, max_length=200)


def owner_required(owner: str | None = Depends(chat_owner)) -> str:
    if owner is None:
        raise HTTPException(status_code=404, detail="Accounts are only used in hosted mode.")
    return owner


@router.get("/account/keys")
def list_keys(owner: str = Depends(owner_required)) -> dict[str, Any]:
    return {"keys": get_service().user_keys.listing(owner)}


@router.put("/account/keys")
def put_key(body: OwnKey, owner: str = Depends(owner_required)) -> dict[str, Any]:
    service = get_service()
    try:
        service.check_user_key(body.provider, body.key)
        service.user_keys.put(owner, body.provider, body.key.strip())
    except KeyRejected as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except KeyStoreError as err:
        raise HTTPException(status_code=503, detail="Own keys cannot be stored right now.") from err
    return {"keys": service.user_keys.listing(owner)}


@router.delete("/account/keys/{provider}")
def delete_key(provider: str, owner: str = Depends(owner_required)) -> dict[str, Any]:
    service = get_service()
    if not service.user_keys.remove(owner, provider):
        raise HTTPException(status_code=404, detail="No key stored for that provider.")
    return {"keys": service.user_keys.listing(owner)}
