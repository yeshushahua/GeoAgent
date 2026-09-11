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
    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_model_path: Path
    vlm_device: Literal["cuda:0"] = "cuda:0"
    vlm_dtype: Literal["bfloat16"] = "bfloat16"
    vlm_attention: Literal["sdpa"] = "sdpa"
    vlm_max_image_edge: int = Field(default=2048, ge=512, le=4096)
    vlm_max_visual_tokens: int = Field(default=1280, ge=256, le=4096)
    vlm_default_max_new_tokens: int = Field(default=256, ge=64, le=512)
    app_name: str = "GeoAgent"
    app_version: str = "0.3.0"
    tool_trace_limit: int = Field(default=50, ge=1, le=1000)
    tool_timeout_seconds: int = Field(default=900, ge=1, le=3600)
    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    gradio_host: str = "127.0.0.1"
    gradio_port: int = Field(default=7860, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_paths(self):
        names = ("project_root", "storage_root", "vlm_model_path", *self.asset_paths.keys())
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
        if not self.vlm_model_path.is_relative_to(self.model_dir) or self.vlm_model_path == self.model_dir:
            raise ValueError("VLM_MODEL_PATH must be a child of MODEL_DIR")
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

    @property
    def vlm_max_pixels(self) -> int:
        return self.vlm_max_visual_tokens * 32 * 32


@lru_cache
def get_settings() -> Settings:
    return Settings()
