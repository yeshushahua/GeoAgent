"""Read local configuration without importing model or UI libraries."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    project_root: Path
    storage_root: Path
    model_dir: Path
    hf_home: Path
    dataset_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    temp_dir: Path
    hf_hub_offline: bool = True
    app_name: str = "GeoAgent"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    gradio_host: str = "127.0.0.1"
    gradio_port: int = Field(default=7860, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_paths(self):
        names = ("project_root", "storage_root", *self.asset_paths.keys())
        for name in names:
            path = getattr(self, name)
            if not path.is_absolute():
                raise ValueError(f"{name.upper()} must be an absolute path")
            setattr(self, name, path.resolve())
        if not self.project_root.is_dir():
            raise ValueError("PROJECT_ROOT must be an existing directory")
        if (self.storage_root == self.project_root
                or self.storage_root.is_relative_to(self.project_root)
                or self.project_root.is_relative_to(self.storage_root)):
            raise ValueError("STORAGE_ROOT and PROJECT_ROOT must be separate directory trees")
        # On Windows enforce the separate-volume policy, including resolved junctions.
        if self.project_root.drive and self.project_root.drive.lower() == self.storage_root.drive.lower():
            raise ValueError("STORAGE_ROOT must be on a different drive from PROJECT_ROOT")
        for name, path in self.asset_paths.items():
            if path == self.storage_root or not path.is_relative_to(self.storage_root):
                raise ValueError(f"{name.upper()} must be a child of STORAGE_ROOT; no fallback allowed")
        if len(set(self.asset_paths.values())) != len(self.asset_paths):
            raise ValueError("Asset directories must be distinct")
        return self

    @property
    def asset_paths(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in (
            "model_dir", "hf_home", "dataset_dir", "output_dir", "checkpoint_dir", "temp_dir"
        )}

    @property
    def api_base_url(self) -> str:
        host = "127.0.0.1" if self.api_host == "0.0.0.0" else self.api_host
        return f"http://{host}:{self.api_port}/api/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
