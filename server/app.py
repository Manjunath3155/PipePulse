from __future__ import annotations

import os

import uvicorn

from app import app


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
