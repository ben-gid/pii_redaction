import os
import secrets
from pathlib import Path
import sys
import boto3
from fastapi import Security, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from .core.config import settings, state
# Ensure the project root is on sys.path so that pii_redaction is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pii_redaction.models import RedactionResponse  # noqa: E402


def get_ip(request: Request) -> str:
    if forwarded := request.headers.get("X-Forwarded-For"):
        return forwarded.split(",")[0].strip()
    client = request.client
    return getattr(client, "host", "unknown") if client else "unknown"

limiter = Limiter(key_func=get_ip)
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(provided_key: str = Security(api_key_header)) -> None:
    if not secrets.compare_digest(provided_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

async def run_redaction(text: str, threshold: float) -> RedactionResponse:
    if state.redactor is None:
        raise HTTPException(status_code=503, detail="Redactor not loaded")
    try:
        return state.redactor.predict(text, threshold)
    except Exception:
        raise HTTPException(status_code=500, detail="Redaction failed during inference")
    
def fetch_and_write_cert() -> tuple[str, str]:
    """
    Fetches the SSL certificate and key from AWS Systems Manager (SSM) 
    Parameter Store and writes them to temporary files.
    
    Returns:
        tuple[str, str]: A tuple containing the paths to the written 
                         certificate and key files respectively.
    """
    ssm = boto3.client("ssm", region_name="us-west-2")
    
    cert = ssm.get_parameter(Name="/pii-redaction/SSL_CERT", WithDecryption=True)["Parameter"]["Value"]
    key = ssm.get_parameter(Name="/pii-redaction/SSL_KEY", WithDecryption=True)["Parameter"]["Value"]
    
    os.makedirs("/tmp/certs", exist_ok=True)
    cert_file = "/tmp/certs/cert.pem"
    key_file = "/tmp/certs/key.pem"
    with open(cert_file, "w") as f:
        f.write(cert)
    with open(key_file, "w") as f:
        f.write(key)
    
    return cert_file, key_file