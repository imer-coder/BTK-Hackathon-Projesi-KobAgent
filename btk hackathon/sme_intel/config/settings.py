"""
config/settings.py — Centralised application configuration for SME-Intel.

Secret resolution priority
---------------------------
1. **Environment variables** — set directly in the OS or in Streamlit Cloud's
   "Secrets" panel (which injects them as ``os.environ`` entries at runtime).
2. **``.env`` file** — for local development only (git-ignored).

This means the SAME code runs both locally (reads ``.env``) and on
Streamlit Community Cloud (reads the Secrets panel) with zero changes.

Usage
-----
    from config.settings import settings

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_FLASH,
        google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
    )

Security note
-------------
``GOOGLE_API_KEY`` and ``DATABASE_URL`` are stored as ``SecretStr``.
Always call ``.get_secret_value()`` when passing them to a library;
never cast to plain ``str`` in logs or error messages.
"""

from __future__ import annotations

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    SME-Intel runtime configuration.

    All fields are resolved from environment variables.
    The ``.env`` file is loaded as a fallback for local development.

    Attributes
    ----------
    GOOGLE_API_KEY      : Google Gemini API key (required).
    DATABASE_URL        : Full database connection string.
                          PostgreSQL in production (set via Streamlit Secrets).
                          Omit locally — SQLite fallback kicks in via database.py.
    GEMINI_MODEL_FLASH  : Fast Gemini model (default: gemini-1.5-flash).
    GEMINI_MODEL_PRO    : Reasoning model (default: gemini-1.5-flash).
    LOG_LEVEL           : Python logging level string (default: INFO).
    APP_TITLE           : Streamlit page title.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Required ─────────────────────────────────────────────────────────── #
    GOOGLE_API_KEY: SecretStr

    # ── Optional — omit locally to use SQLite fallback in database.py ────── #
    DATABASE_URL: SecretStr | None = None

    # ── LLM model names ──────────────────────────────────────────────────── #
    GEMINI_MODEL_FLASH: str = "gemini-1.5-flash-latest"
    GEMINI_MODEL_PRO: str = "gemini-1.5-flash-latest"

    # ── Misc ─────────────────────────────────────────────────────────────── #
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "SME-Intel (KOBİ-Zeka)"

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got {v!r}.")
        return v.upper()


# Module-level singleton — import this everywhere.
settings: Settings = Settings()
