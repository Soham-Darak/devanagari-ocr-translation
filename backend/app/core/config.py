
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Lekha API"
    app_env: str = "development"
    api_prefix: str = "/api"

    frontend_origin: str = "http://localhost:5173"

    model_checkpoint: str = "./models/devanagari_cnn_best.pth"
    model_device: str = "auto"

    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    max_pdf_pages: int = Field(default=20, ge=1, le=100)
    pdf_render_dpi: int = Field(default=180, ge=72, le=300)

    google_translate_api_key: str = ""
    google_translate_timeout_seconds: float = Field(
        default=20.0, gt=0, le=120
    )
    max_translation_characters: int = Field(
        default=30000, ge=1, le=100000
    )

    @property
    def checkpoint_path(self) -> Path:
        path = Path(self.model_checkpoint)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path.resolve()

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip().rstrip("/")
            for origin in self.frontend_origin.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
