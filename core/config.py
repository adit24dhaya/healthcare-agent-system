import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_token: str
    require_api_token: bool
    request_timeout_seconds: int
    log_dir: Path


def get_settings():
    project_root = Path(__file__).resolve().parents[1]
    return Settings(
        api_token=os.getenv("API_TOKEN", ""),
        require_api_token=_as_bool(os.getenv("REQUIRE_API_TOKEN"), default=False),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "25")),
        log_dir=Path(os.getenv("LOG_DIR", str(project_root / "logs"))),
    )
