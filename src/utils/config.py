"""Configuration management for environment variables."""

from dotenv import load_dotenv
import os
from pathlib import Path


def load_env() -> bool:
    """
    Load environment variables from .env file.

    Locates .env in the project root (two levels above src/utils/)
    and loads HF_TOKEN into os.environ.

    Returns:
        bool: True if HF_TOKEN is loaded, False otherwise.
    """
    # Locate .env in project root: src/utils/config.py → src → . (root)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        return True
    return False
