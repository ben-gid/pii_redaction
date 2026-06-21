from pathlib import Path
from dotenv import load_dotenv
import sys
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pii_redaction.redactor import PIIRedactor  # noqa: E402

load_dotenv()

class AppSettings(BaseSettings):
    model_id: str = "bengid/pii-redaction-deberta-small"
    threshold: float = 0.85
    api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env")

class AppState:
    redactor: Optional[PIIRedactor] = None

    def load(self, settings: AppSettings) -> None:
        self.redactor = PIIRedactor(model_id=settings.model_id)

    def clear(self) -> None:
        self.redactor = None

settings = AppSettings()
state = AppState()