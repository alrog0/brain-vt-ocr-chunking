"""
OCR + chunking + embeddings orchestrator (FastAPI/OpenAPI, psycopg2, no plpy).

Main goals:
- Single service file at project root.
- Required input: oid (PostgreSQL Large Object OID del archivo).
- Full document processing or first/last N pages.
- Queue + overwrite policies with clear and traceable states.
- Docling OCR, semantic/simple chunking, embedding generation, DB persistence.
- Mock mode for local tests without DB/server.

Run API server:
  python ocr_chunking.py --host 0.0.0.0 --port 8000

Run local mock demo (no DB):
  python ocr_chunking.py --mock-local
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import hashlib
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import fitz
import jwt
import psycopg2
import torch
import torch.nn.functional as torch_functional
import uvicorn
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings as docling_settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.jwks_client import PyJWKClient
from jwt.exceptions import ExpiredSignatureError, InvalidIssuerError, InvalidTokenError
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor, execute_batch
from transformers import AutoModel, AutoTokenizer


SERVICE_NAME = "OCR Chunking Embeddings Orchestrator"
SERVICE_VERSION = "3.0.0-20260323"
TAG_CHUNKING = "Segmento Chunking"
TAG_EMBEDDING = "Segmento Embedding"
TAG_OCR = "Segmento Docling-OCR"
TAG_PIPELINE = "Segmento Pipeline"
TAG_HELPERS = "Segmento Helpers"
TAG_INFRA = "Segmento Infraestructura"
TAG_VALIDATION = "Segmento Validacion Integral"
TAG_LOGS = "Segmento Logs"

OPENAPI_TAGS = [
    {"name": TAG_CHUNKING, "description": "Metodos de chunking (texto a chunks)."},
    {"name": TAG_EMBEDDING, "description": "Metodos de generacion de embeddings."},
    {"name": TAG_OCR, "description": "Metodos de OCR y extraccion de texto."},
    {"name": TAG_PIPELINE, "description": "Metodos de orquestacion completa PipelineOCR."},
    {"name": TAG_HELPERS, "description": "Endpoints auxiliares: auth/login, health, example-request, validate-db."},
    {"name": TAG_INFRA, "description": "Diagnostico de infraestructura: GPU, librerias, compatibilidad CUDA."},
    {"name": TAG_VALIDATION, "description": "Validacion integral del pipeline con upload de archivos de prueba."},
    {"name": TAG_LOGS, "description": "Consulta de logs estructurados del servicio."},
]

DEFAULT_QUEUE_NAME = "BRAINVT_OCR_EMBEDDINGS_GPU"
DEFAULT_JOB_TYPE = "BRAINVT_OCR_EMBEDDINGS_GPU"
DEFAULT_CREATED_BY = 1101
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_PROBE_MAX_PAGES = 60
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_AUTH_ENABLED = True
DEFAULT_JWT_ISSUER = "https://anh-pro.flows.ninja/auth/realms/airflows"
DEFAULT_JWKS_URL = f"{DEFAULT_JWT_ISSUER}/protocol/openid-connect/certs"
DEFAULT_JWT_AZP = "ocr-chunking-embeddings"
DEFAULT_JWT_CLIENT_ID = "ocr-chunking-embeddings"
SERVICE_STAGE_ENDPOINTS = {
    "ocr": "/ocr-docling/process",
    "chunking": "/chunking-docling/process",
    "embedding": "/embedding-generation/process",
    "pipeline": "/PipelineOCR/process",
}
# ---------------------------------------------------------------------------
# Supported formats — aligned with docling 2.81 InputFormat enum (17 formats)
# ---------------------------------------------------------------------------
SUPPORTED_DOCLING_MIME_TYPES = {
    # PDF
    "application/pdf",
    # Office Open XML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Legacy Office (OLE2) — se convierten a OOXML con LibreOffice en runtime
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    # Texto plano
    "text/plain",
    # Markup / text
    "text/markdown",
    "text/x-markdown",
    "text/asciidoc",
    "text/x-asciidoc",
    "text/x-tex",
    "application/x-latex",
    "text/html",
    "application/xhtml+xml",
    "text/csv",
    # XML especializados (Docling soporta USPTO, JATS, XBRL)
    "application/xml",
    "text/xml",
    "application/vnd.webvtt",
    "text/vtt",
    # Docling JSON
    "application/json",
    # Images
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
    # Audio (requires docling[asr])
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    # Subtitles
    "text/vtt",
    # XML schemas
    "application/xml",
    "text/xml",
    # Docling JSON (re-import)
    "application/json",
}
MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/asciidoc": ".adoc",
    "text/x-asciidoc": ".adoc",
    "text/x-tex": ".tex",
    "application/x-latex": ".tex",
    "text/html": ".html",
    "application/xhtml+xml": ".xhtml",
    "text/csv": ".csv",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "text/vtt": ".vtt",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/json": ".json",
}
EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",  # legacy OLE2 → se convierte a docx en runtime
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",  # legacy OLE2 → se convierte a xlsx en runtime
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",  # legacy OLE2 → se convierte a pptx en runtime
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".adoc": "text/asciidoc",
    ".asciidoc": "text/asciidoc",
    ".tex": "text/x-tex",
    ".latex": "text/x-tex",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".vtt": "text/vtt",
    ".xml": "application/xml",
    ".json": "application/json",
    ".rtf": "text/html",  # RTF → treat as HTML for extraction
}
# Magic bytes for binary MIME detection fallback
_MAGIC_SIGNATURES = [
    (b"%PDF", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # ZIP-based: docx, xlsx, pptx
    (b"\xd0\xcf\x11\xe0", "application/msword"),  # OLE2: legacy doc/xls/ppt
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
    (b"RIFF", "audio/wav"),
    (b"II\x2a\x00", "image/tiff"),
    (b"MM\x00\x2a", "image/tiff"),
]

PAGE_INDICATOR_PATTERN = re.compile(
    r"(?i)^\s*(?:pagina\s+\d+\s+de\s+\d+|page\s+\d+|\-\s*\d+\s*\-|\[\d+\])\s*$",
    re.MULTILINE,
)
ISOLATED_NUMBER_PATTERN = re.compile(r"^[\d\s\.,\-/]+$")
MULTI_EMPTY_LINES_PATTERN = re.compile(r"\n{3,}")
WORD_TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÜüÑñ]+", re.UNICODE)

# ---------------------------------------------------------------------------
# Procesamiento de documentos grandes: divide PDFs en bloques de N paginas
# para evitar std::bad_alloc en docling-parse y habilitar GC entre bloques.
# Env OCR_PAGE_CHUNK_SIZE=0 desactiva el chunking (procesa el doc completo).
# ---------------------------------------------------------------------------
PAGE_CHUNK_SIZE = int(os.getenv("OCR_PAGE_CHUNK_SIZE", "50"))
PAGE_CHUNK_SIZE_HEAVY = int(os.getenv("OCR_PAGE_CHUNK_SIZE_HEAVY", "25"))
PAGE_CHUNK_DOCUMENT_TIMEOUT = float(os.getenv("OCR_DOCUMENT_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Timeouts adaptativos por tipo MIME (segundos).
# El consumer usa estos valores para estimar si un batch excede el timeout global.
# ---------------------------------------------------------------------------
TIMEOUT_PER_FORMAT = {
    "application/pdf": 3.0,           # ~3s por pagina (OCR + layout)
    "application/msword": 15.0,       # Conversion LibreOffice + parsing
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 10.0,
    "application/vnd.ms-excel": 20.0,  # Puede ser lento en hojas grandes
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 15.0,
    "application/vnd.ms-powerpoint": 15.0,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": 10.0,
    "image/jpeg": 5.0,
    "image/png": 5.0,
    "image/tiff": 8.0,
    "image/bmp": 5.0,
    "text/plain": 2.0,
    "text/csv": 2.0,
    "text/html": 3.0,
    "application/json": 2.0,
    "application/xml": 3.0,
}
TIMEOUT_EMBEDDINGS_BASE = 10.0  # Tiempo base para generacion de embeddings por documento


def estimate_document_timeout(mime_type: str, pages: int = 1, size_bytes: int = 0) -> float:
    """Estima el tiempo de procesamiento (segundos) para un documento.

    El consumer usa esta funcion para decidir la composicion del batch
    y evitar exceder el timeout global.
    Formula: base_por_formato * paginas + embeddings_base + penalizacion_tamano
    """
    base = TIMEOUT_PER_FORMAT.get(normalize_mime_type(mime_type), 5.0)
    # PDFs: escala por paginas. Otros: base fija.
    if "pdf" in (mime_type or "").lower():
        estimated = base * max(pages, 1) + TIMEOUT_EMBEDDINGS_BASE
    else:
        estimated = base + TIMEOUT_EMBEDDINGS_BASE
    # Penalizacion por tamano: archivos >50MB reciben tiempo extra
    if size_bytes > 50_000_000:
        estimated += (size_bytes / 50_000_000) * 30.0
    return estimated


def estimate_batch_timeout(items: List[Dict[str, Any]]) -> float:
    """Estima el tiempo total para un batch de documentos.

    Cada item debe tener: mime_type, pages (opcional), size_bytes (opcional).
    Retorna el total estimado en segundos con 20% de margen de seguridad.
    """
    total = 0.0
    for item in items:
        total += estimate_document_timeout(
            item.get("mime_type", "application/pdf"),
            item.get("pages", 1),
            item.get("size_bytes", 0),
        )
    return total * 1.2  # 20% margen de seguridad

# Configure docling global performance settings for GPU throughput.
# page_batch_size controls how many pages are sent to GPU models at once.
docling_settings.perf.page_batch_size = 64
docling_settings.perf.page_batch_concurrency = 2

LOGGER = logging.getLogger("ocr_chunking")
if not LOGGER.handlers:
    logging.basicConfig(
        level=os.getenv("OCR_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_DOCLING_CONVERTER_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = Lock()
_TOKENIZER_LOCK = Lock()
_DOCLING_LOCK = Lock()
_JWK_CLIENT: Optional[PyJWKClient] = None
_AUTH_SETTINGS_CACHE: Optional["AuthSettings"] = None
_PG_SETTINGS_CACHE: Optional["PostgresSettings"] = None
_BEARER_SCHEME = HTTPBearer(
    auto_error=False,
    description="JWT Bearer emitido por Keycloak para acceder al servicio.",
)


@dataclass
class AuthSettings:
    """JWT auth settings for API access control."""

    enabled: bool
    issuer: str
    jwks_url: str
    audience: Optional[str]
    expected_azp: Optional[str]
    expected_client_id: Optional[str]
    algorithms: List[str]

    @staticmethod
    def from_env() -> "AuthSettings":
        """Loads JWT auth settings from environment (cached after first call)."""
        global _AUTH_SETTINGS_CACHE
        if _AUTH_SETTINGS_CACHE is not None:
            return _AUTH_SETTINGS_CACHE
        algorithms_raw = safe_str(os.getenv("OCR_JWT_ALGORITHMS", "RS256"), "RS256")
        algorithms = [item.strip() for item in algorithms_raw.split(",") if item.strip()]
        _AUTH_SETTINGS_CACHE = AuthSettings(
            enabled=safe_bool(os.getenv("OCR_AUTH_ENABLED", str(DEFAULT_AUTH_ENABLED).lower()), DEFAULT_AUTH_ENABLED),
            issuer=safe_str(os.getenv("OCR_JWT_ISSUER", DEFAULT_JWT_ISSUER), DEFAULT_JWT_ISSUER).strip(),
            jwks_url=safe_str(os.getenv("OCR_JWKS_URL", DEFAULT_JWKS_URL), DEFAULT_JWKS_URL).strip(),
            audience=safe_str(os.getenv("OCR_JWT_AUDIENCE", ""), "").strip() or None,
            expected_azp=safe_str(os.getenv("OCR_JWT_EXPECTED_AZP", DEFAULT_JWT_AZP), DEFAULT_JWT_AZP).strip() or None,
            expected_client_id=safe_str(
                os.getenv("OCR_JWT_EXPECTED_CLIENT_ID", DEFAULT_JWT_CLIENT_ID),
                DEFAULT_JWT_CLIENT_ID,
            ).strip()
            or None,
            algorithms=algorithms or ["RS256"],
        )
        return _AUTH_SETTINGS_CACHE

def utc_now_iso() -> str:
    """Returns UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_str(value: Any, default: str = "") -> str:
    """Safe str conversion."""
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safe int conversion."""
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safe float conversion."""
    try:
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safe bool conversion."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = safe_str(value, "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "si", "s", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return bool(default)


def to_json_dict(value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Converts value to dict when possible."""
    if default is None:
        default = {}
    if isinstance(value, dict):
        return value
    if value is None:
        return dict(default)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return dict(default)
    return dict(default)


def to_json_safe(value: Any) -> Any:
    """Converts values to JSON-safe structures (datetime -> ISO, etc.)."""
    if isinstance(value, dict):
        return {safe_str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, set):
        return [to_json_safe(v) for v in sorted(value, key=lambda x: safe_str(x))]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return safe_str(value, "")
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return safe_str(value, "")
    return value


def json_dumps_safe(value: Any) -> str:
    """JSON dump that never fails on datetime/date/bytes."""
    return json.dumps(to_json_safe(value), ensure_ascii=False)


def normalize_file_name(value: Any) -> str:
    """Normaliza nombre de archivo a basename limpio."""
    raw = safe_str(value, "").strip()
    if not raw:
        return ""
    return os.path.basename(raw.replace("\\", "/")).strip()


def normalize_mime_type(value: Any) -> str:
    """Normaliza mime type a minusculas, sin parametros."""
    raw = safe_str(value, "").strip().lower()
    if not raw:
        return ""
    if ";" in raw:
        raw = raw.split(";", 1)[0].strip()
    return raw


def infer_mime_type_from_file_name(file_name: str) -> str:
    """Infiere mime type por extension de archivo."""
    normalized = normalize_file_name(file_name)
    if not normalized:
        return ""
    _, ext = os.path.splitext(normalized)
    return EXTENSION_TO_MIME.get(ext.lower(), "")


def infer_mime_from_binary(data: bytes) -> str:
    """Detecta MIME type por magic bytes del contenido binario."""
    if not data or len(data) < 4:
        return ""
    header = data[:8]
    for signature, mime in _MAGIC_SIGNATURES:
        if header[:len(signature)] == signature:
            return mime
    return ""


def resolve_zip_based_mime(data: bytes, file_name: str) -> str:
    """Para archivos ZIP (docx/xlsx/pptx), resuelve el tipo real por extensión o contenido."""
    ext = os.path.splitext(normalize_file_name(file_name))[1].lower() if file_name else ""
    if ext in (".docx", ".doc"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext in (".xlsx", ".xls"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if ext in (".pptx", ".ppt"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    # Intentar detectar por contenido del ZIP
    try:
        import zipfile
        import io
        with zipfile.ZipFile(io.BytesIO(data[:65536])) as zf:
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if "xl/workbook.xml" in names:
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if "ppt/presentation.xml" in names:
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    except Exception:
        pass
    return "application/zip"


def resolve_request_mime_type(
    request_mime_type: Optional[str],
    request_metadata: Dict[str, Any],
    documento_info: Optional[Dict[str, Any]],
    file_name: str,
    binary_data: Optional[bytes] = None,
) -> str:
    """Resuelve mime type desde request, metadata, documento, extension y contenido binario."""
    candidate = normalize_mime_type(request_mime_type)
    if candidate:
        return candidate
    candidate = normalize_mime_type(request_metadata.get("mime_type"))
    if candidate:
        return candidate
    candidate = normalize_mime_type(request_metadata.get("mimeType"))
    if candidate:
        return candidate
    candidate = normalize_mime_type((documento_info or {}).get("archivo_mime_type"))
    if candidate:
        return candidate
    candidate = normalize_mime_type(infer_mime_type_from_file_name(file_name))
    if candidate:
        return candidate
    # Fallback: detección por contenido binario
    if binary_data:
        magic_mime = infer_mime_from_binary(binary_data)
        if magic_mime == "application/zip":
            return resolve_zip_based_mime(binary_data, file_name)
        if magic_mime == "application/msword":
            # Legacy OLE2 — map by extension or default to docx
            ext = os.path.splitext(normalize_file_name(file_name))[1].lower() if file_name else ""
            return EXTENSION_TO_MIME.get(ext, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        if magic_mime:
            return magic_mime
    return ""


def is_supported_docling_mime(mime_type: str) -> bool:
    """Valida si mime type esta soportado por el servicio Docling actual."""
    return normalize_mime_type(mime_type) in SUPPORTED_DOCLING_MIME_TYPES


def pydantic_model_dump(model: BaseModel) -> Dict[str, Any]:
    """Serializes Pydantic model to dict."""
    return model.model_dump()


def pydantic_model_dump_json(model: BaseModel, *, indent: int = 2) -> str:
    """Serializes Pydantic model to JSON string."""
    return model.model_dump_json(indent=indent)

def _auth_error_detail(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Builds standardized auth error payload."""
    return {
        "status": "FAILED",
        "exitoso": False,
        "message": "Acceso denegado.",
        "error": {
            "phase": "AUTH",
            "code": code,
            "message": message,
            "details": details or {},
        },
        "timestamp_utc": utc_now_iso(),
    }


def _get_jwk_client(settings: AuthSettings) -> PyJWKClient:
    """Returns a cached JWKS client for the configured issuer."""
    global _JWK_CLIENT
    if _JWK_CLIENT is None:
        _JWK_CLIENT = PyJWKClient(settings.jwks_url)
    return _JWK_CLIENT


def _extract_bearer_token(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """Extracts Bearer token from FastAPI security credentials."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_REQUIRED",
                    message="Debe enviar Authorization: Bearer <token>.",
                )
            ),
        )

    scheme = safe_str(getattr(credentials, "scheme", ""), "").strip().lower()
    token = safe_str(getattr(credentials, "credentials", ""), "").strip()
    if scheme != "bearer":
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_INVALID_HEADER",
                    message="El header Authorization debe ser Bearer <token>.",
                )
            ),
        )

    if not token:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_EMPTY_TOKEN",
                    message="Token Bearer vacio.",
                )
            ),
        )
    return token


def validate_keycloak_token(token: str, settings: AuthSettings) -> Dict[str, Any]:
    """Validates JWT signature and required claims against Keycloak JWKS."""
    try:
        signing_key = _get_jwk_client(settings).get_signing_key_from_jwt(token)
        decode_kwargs: Dict[str, Any] = {
            "jwt": token,
            "key": signing_key.key,
            "algorithms": settings.algorithms,
            "issuer": settings.issuer,
            "options": {
                "require": ["exp", "iat", "iss"],
                "verify_aud": settings.audience is not None,
            },
        }
        if settings.audience is not None:
            decode_kwargs["audience"] = settings.audience

        payload = jwt.decode(**decode_kwargs)

        if settings.expected_azp and safe_str(payload.get("azp"), "").strip() != settings.expected_azp:
            raise HTTPException(
                status_code=403,
                detail=to_json_safe(
                    _auth_error_detail(
                        code="AUTH_INVALID_AZP",
                        message="Claim azp no autorizado.",
                        details={"expected": settings.expected_azp, "received": payload.get("azp")},
                    )
                ),
            )

        if (
            settings.expected_client_id
            and safe_str(payload.get("clientId"), "").strip() != settings.expected_client_id
        ):
            raise HTTPException(
                status_code=403,
                detail=to_json_safe(
                    _auth_error_detail(
                        code="AUTH_INVALID_CLIENT_ID",
                        message="Claim clientId no autorizado.",
                        details={
                            "expected": settings.expected_client_id,
                            "received": payload.get("clientId"),
                        },
                    )
                ),
            )
        return payload
    except HTTPException:
        raise
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_TOKEN_EXPIRED",
                    message="Token expirado.",
                )
            ),
        )
    except InvalidIssuerError:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_INVALID_ISSUER",
                    message="Issuer invalido.",
                )
            ),
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_INVALID_TOKEN",
                    message="Token invalido.",
                    details={"reason": safe_str(exc)},
                )
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=to_json_safe(
                _auth_error_detail(
                    code="AUTH_VALIDATION_ERROR",
                    message="No fue posible validar el token.",
                    details={"reason": safe_str(exc)},
                )
            ),
        )


def require_service_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_BEARER_SCHEME),
) -> Dict[str, Any]:
    """Mandatory API auth via JWT Bearer token validated against Keycloak JWKS."""
    settings = AuthSettings.from_env()
    if not settings.enabled:
        return {}
    token = _extract_bearer_token(credentials)
    return validate_keycloak_token(token, settings)


def vector_to_pg_literal(vector: List[float]) -> str:
    """Converts vector to pgvector literal."""
    return "[" + ",".join(f"{float(x):.10f}" for x in vector) + "]"


def mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9,
    )


class PipelineError(Exception):
    """Pipeline exception with phase and machine-friendly error code."""

    def __init__(self, phase: str, code: str, message: str,
                 details: Optional[Dict[str, Any]] = None,
                 retryable: bool = True) -> None:
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error."""
        return {
            "phase": self.phase,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class PhaseRecorder:
    """Trace recorder for all pipeline phases."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def push(self, phase: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adds one phase event."""
        event = {
            "phase": phase,
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp_utc": utc_now_iso(),
        }
        self.events.append(event)
        return event

    def as_list(self) -> List[Dict[str, Any]]:
        """Returns all events."""
        return list(self.events)


@dataclass
class PostgresSettings:
    """Postgres settings."""

    host: str
    port: int
    dbname: str
    user: str
    password: str

    @staticmethod
    def from_env() -> "PostgresSettings":
        """Loads DB settings from environment (cached after first call)."""
        global _PG_SETTINGS_CACHE
        if _PG_SETTINGS_CACHE is not None:
            return _PG_SETTINGS_CACHE
        _PG_SETTINGS_CACHE = PostgresSettings(
            host=os.getenv("OCR_DB_HOST", "localhost"),
            port=int(os.getenv("OCR_DB_PORT", "5432")),
            dbname=os.getenv("OCR_DB_NAME", "niledb"),
            user=os.getenv("OCR_DB_USER", "postgres"),
            password=os.getenv("OCR_DB_PASSWORD", "plexia"),
        )
        return _PG_SETTINGS_CACHE


def bogota_now_iso() -> str:
    """Returns current time in America/Bogota offset as ISO string."""
    bogota_tz = timezone(timedelta(hours=-5))
    return datetime.now(bogota_tz).replace(microsecond=0).isoformat()


def _validate_db_connection() -> Dict[str, Any]:
    """
    Valida conexión a Postgres.
    Retorna versión, metadatos y resultado de una consulta de prueba.
    """
    settings = PostgresSettings.from_env()
    try:
        conn = psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.dbname,
            user=settings.user,
            password=settings.password,
            connect_timeout=5,
        )
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT current_database() AS current_database, current_user AS current_user, version() AS version;"
                )
                row = cur.fetchone()
                if not row:
                    return {
                        "status": "error",
                        "message": "No se obtuvo fila de metadatos.",
                        "postgres_version": None,
                        "metadata": None,
                        "test_query": None,
                        "error": "Empty result",
                    }
                version_str = row["version"] or ""
                version_short = version_str.split(",")[0].strip() if version_str else None
                metadata = {
                    "current_database": row["current_database"],
                    "current_user": row["current_user"],
                    "server_version_full": version_str,
                    "server_version_short": version_short,
                    "connection": {
                        "host": settings.host,
                        "port": settings.port,
                        "dbname": settings.dbname,
                        "user": settings.user,
                    },
                }
                cur.execute(
                    "SELECT 1 AS ping, current_timestamp AS server_time, "
                    "current_setting('server_version_num') AS version_num;"
                )
                test_row = cur.fetchone()
                test_query = dict(test_row) if test_row else None
            return {
                "status": "ok",
                "message": "Conexión exitosa.",
                "postgres_version": version_short,
                "metadata": metadata,
                "test_query": test_query,
                "error": None,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {
            "status": "error",
            "message": "No se pudo conectar a Postgres.",
            "postgres_version": None,
            "metadata": {
                "connection": {
                    "host": settings.host,
                    "port": settings.port,
                    "dbname": settings.dbname,
                }
            },
            "test_query": None,
            "error": str(exc),
        }


class PostgresClient:
    """Postgres adapter for queue/job/embedding operations."""

    def __init__(self, settings: PostgresSettings) -> None:
        self.settings = settings
        self._connect()

    def _connect(self) -> None:
        self.conn = psycopg2.connect(
            host=self.settings.host,
            port=self.settings.port,
            dbname=self.settings.dbname,
            user=self.settings.user,
            password=self.settings.password,
        )
        self.conn.autocommit = False

    def _is_alive(self) -> bool:
        try:
            if self.conn.closed:
                return False
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _ensure_connection(self) -> None:
        if not self._is_alive():
            try:
                self.conn.close()
            except Exception:
                pass
            self._connect()

    def _reconnect(self) -> None:
        """Force reconnect after connection loss."""
        try:
            self.conn.close()
        except Exception:
            pass
        import time
        for attempt in range(3):
            try:
                self._connect()
                LOGGER.info("PostgresClient reconnected (attempt %d)", attempt + 1)
                return
            except Exception as exc:
                if attempt < 2:
                    wait = (attempt + 1) * 2
                    LOGGER.warning(
                        "Reconnect attempt %d failed: %s. Retrying in %ds...",
                        attempt + 1, str(exc)[:200], wait,
                    )
                    time.sleep(wait)
                else:
                    LOGGER.error("Reconnect failed after 3 attempts: %s", str(exc)[:200])
                    raise

    def _is_connection_error(self, exc: Exception) -> bool:
        """Check if exception is a connection-level error (recoverable by reconnect)."""
        return isinstance(exc, (
            psycopg2.OperationalError,
            psycopg2.InterfaceError,
        ))

    def _safe_rollback(self) -> None:
        try:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
        except Exception:
            pass

    def __enter__(self) -> "PostgresClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self._safe_rollback()
        try:
            self.conn.close()
        except Exception:
            pass

    def query_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Executes SELECT and returns one dict row. Retries once on connection error."""
        for attempt in range(2):
            self._ensure_connection()
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                return dict(row) if row is not None else None
            except Exception as exc:
                if attempt == 0 and self._is_connection_error(exc):
                    LOGGER.warning("query_one connection lost, reconnecting...")
                    self._reconnect()
                    continue
                raise

    def query_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """Executes SELECT and returns all dict rows. Retries once on connection error."""
        for attempt in range(2):
            self._ensure_connection()
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                return [dict(r) for r in rows]
            except Exception as exc:
                if attempt == 0 and self._is_connection_error(exc):
                    LOGGER.warning("query_all connection lost, reconnecting...")
                    self._reconnect()
                    continue
                raise

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> int:
        """Executes write SQL and returns affected rows. Retries once on connection error."""
        for attempt in range(2):
            self._ensure_connection()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(sql, params)
                    affected = int(cur.rowcount)
                self.conn.commit()
                return affected
            except Exception as exc:
                self._safe_rollback()
                if attempt == 0 and self._is_connection_error(exc):
                    LOGGER.warning("execute connection lost, reconnecting...")
                    self._reconnect()
                    continue
                raise

    def execute_returning_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        """Executes write SQL with RETURNING and returns one row. Retries once on connection error."""
        for attempt in range(2):
            self._ensure_connection()
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                self.conn.commit()
                return dict(row) if row is not None else None
            except Exception as exc:
                self._safe_rollback()
                if attempt == 0 and self._is_connection_error(exc):
                    LOGGER.warning("execute_returning_one connection lost, reconnecting...")
                    self._reconnect()
                    continue
                raise

    def fetch_item_by_oid(self, oid: int) -> Optional[Dict[str, Any]]:
        """Gets latest item from Operaciones.ItemsIngestaSmb by loOid."""
        sql = """
        SELECT
          i."trabajoId"      AS trabajo_id,
          i.id               AS item_id,
          i."nombreArchivo"  AS nombre_archivo,
          i.estado           AS estado,
          i."loOid"          AS lo_oid,
          i."bytesEscritos"  AS bytes_escritos,
          i."tamanoBytes"    AS tamano_bytes,
          i."hashSha256"     AS sha256,
          i."cargadoEn"      AS cargado_en
        FROM "Operaciones"."ItemsIngestaSmb" i
        WHERE i."loOid" = %s
        ORDER BY i."trabajoId" DESC, i.id DESC
        LIMIT 1
        """
        return self.query_one(sql, (int(oid),))

    def fetch_documento_by_metadata_oid(self, oid: int) -> Optional[Dict[str, Any]]:
        """Resuelve GestorDocumental.Documentos por metadatosExtra.ocr.metadata.oid."""
        sql_json = """
        SELECT
          d."id"           AS documento_id,
          d."archivoNombre" AS archivo_nombre,
          d."archivoMimeType" AS archivo_mime_type,
          d."contenidoHash" AS contenido_hash,
          d."estado"        AS estado_documento,
          d."procesado"     AS procesado,
          d."embeddingGenerado" AS embedding_generado,
          d."createdBy"     AS created_by,
          d."updatedAt"     AS updated_at
        FROM "GestorDocumental"."Documentos" d
        WHERE COALESCE((d."metadatosExtra"::jsonb -> 'ocr' -> 'metadata' ->> 'oid'), '') ~ '^[0-9]+$'
          AND ((d."metadatosExtra"::jsonb -> 'ocr' -> 'metadata' ->> 'oid')::bigint = %s)
        ORDER BY d."updatedAt" DESC NULLS LAST, d."id" DESC
        LIMIT 1
        """
        try:
            return self.query_one(sql_json, (int(oid),))
        except Exception:
            # Fallback robusto cuando el tipo/estructura de metadatosExtra no permite cast a jsonb.
            pattern = f'"oid"\\s*:\\s*{int(oid)}(?:[^0-9]|$)'
            sql_text = """
            SELECT
              d."id"            AS documento_id,
              d."archivoNombre" AS archivo_nombre,
              d."archivoMimeType" AS archivo_mime_type,
              d."contenidoHash" AS contenido_hash,
              d."estado"        AS estado_documento,
              d."procesado"     AS procesado,
              d."embeddingGenerado" AS embedding_generado,
              d."createdBy"     AS created_by,
              d."updatedAt"     AS updated_at
            FROM "GestorDocumental"."Documentos" d
            WHERE COALESCE(d."metadatosExtra"::text, '') ~ %s
            ORDER BY d."updatedAt" DESC NULLS LAST, d."id" DESC
            LIMIT 1
            """
            return self.query_one(sql_text, (pattern,))

    def fetch_documento_by_file_name(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Resuelve GestorDocumental.Documentos por archivoNombre (fallback operativo)."""
        normalized = normalize_file_name(file_name)
        if not normalized:
            return None
        sql = """
        SELECT
          d."id"            AS documento_id,
          d."archivoNombre" AS archivo_nombre,
          d."archivoMimeType" AS archivo_mime_type,
          d."contenidoHash" AS contenido_hash,
          d."estado"        AS estado_documento,
          d."procesado"     AS procesado,
          d."embeddingGenerado" AS embedding_generado,
          d."createdBy"     AS created_by,
          d."updatedAt"     AS updated_at
        FROM "GestorDocumental"."Documentos" d
        WHERE lower(COALESCE(d."archivoNombre", '')) = lower(%s)
           OR lower(COALESCE(d."archivoNombre", '')) = lower(%s)
        ORDER BY d."updatedAt" DESC NULLS LAST, d."id" DESC
        LIMIT 1
        """
        return self.query_one(sql, (normalized, safe_str(file_name, "").strip()))

    def fetch_large_object_stats(self, oid: int) -> Optional[Dict[str, Any]]:
        """Obtiene paginas/bytes aproximados del pg_largeobject para trazabilidad."""
        sql = """
        SELECT
          l.loid AS oid,
          COUNT(*)::int AS paginas,
          ((MAX(l.pageno) + 1) * 2048)::bigint AS bytes_aprox
        FROM pg_largeobject l
        WHERE l.loid = %s
        GROUP BY l.loid
        """
        return self.query_one(sql, (int(oid),))

    def update_documento_ocr_text(
        self,
        documento_id: int,
        contenido_texto: str,
        contenido_hash: str,
        calidad_ocr: Optional[float],
        paginas: Optional[int],
        palabras: int,
        updated_by: Optional[int],
        estado: str = "EN_PROCESAMIENTO",
    ) -> Optional[Dict[str, Any]]:
        """Actualiza contenido OCR en GestorDocumental.Documentos."""
        sql = """
        UPDATE "GestorDocumental"."Documentos"
        SET
          "contenidoTexto" = %s,
          "contenidoHash" = %s,
          "ocrAplicado" = true,
          "faseActual" = CASE WHEN "faseActual" IN ('INGESTA', 'OCR') THEN 'OCR' ELSE "faseActual" END,
          "calidadOcr" = COALESCE(%s, "calidadOcr"),
          "paginas" = COALESCE(%s, "paginas"),
          "palabras" = %s,
          "estado" = %s,
          "updatedAt" = (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
          "updatedBy" = COALESCE(%s, "updatedBy")
        WHERE "id" = %s
        RETURNING
          "id" AS documento_id,
          "archivoNombre" AS archivo_nombre,
          "estado" AS estado_documento,
          "ocrAplicado" AS ocr_aplicado,
          "calidadOcr" AS calidad_ocr,
          "paginas" AS paginas,
          "palabras" AS palabras,
          "updatedAt" AS updated_at
        """
        return self.execute_returning_one(
            sql,
            (
                safe_str(contenido_texto, ""),
                safe_str(contenido_hash, ""),
                safe_float(calidad_ocr, None),
                safe_int(paginas, None),
                int(max(0, safe_int(palabras, 0) or 0)),
                safe_str(estado, "EN_PROCESAMIENTO"),
                safe_int(updated_by, None),
                int(documento_id),
            ),
        )

    def update_documento_embedding_completion(
        self,
        documento_id: int,
        metadata_patch: Dict[str, Any],
        updated_by: Optional[int],
        estado: str = "EN_PROCESAMIENTO",
    ) -> Optional[Dict[str, Any]]:
        """Actualiza metadatosExtra tras embeddings/chunking.

        NO modifica embeddingGenerado ni estado — eso lo hace el consumer
        (consume_jobs_brainvt_gd) que tiene la vision completa del pipeline.
        """
        sql = """
        UPDATE "GestorDocumental"."Documentos"
        SET
          "metadatosExtra" = COALESCE("metadatosExtra"::jsonb, '{}'::jsonb) || %s::jsonb,
          "updatedAt" = (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
          "updatedBy" = COALESCE(%s, "updatedBy")
        WHERE "id" = %s
        RETURNING
          "id" AS documento_id,
          "archivoNombre" AS archivo_nombre,
          "estado" AS estado_documento,
          "embeddingGenerado" AS embedding_generado,
          "updatedAt" AS updated_at
        """
        return self.execute_returning_one(
            sql,
            (
                json_dumps_safe(metadata_patch),
                safe_int(updated_by, None),
                int(documento_id),
            ),
        )

    def count_processed_documents_by_hash(
        self,
        contenido_hash: str,
        exclude_documento_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Cuenta documentos PROCESADOS con mismo hash."""
        sql = """
        SELECT
          COUNT(*)::int AS total,
          COALESCE(jsonb_agg(d."id") FILTER (WHERE d."id" IS NOT NULL), '[]'::jsonb) AS documento_ids
        FROM "GestorDocumental"."Documentos" d
        WHERE COALESCE(d."contenidoHash",'') = %s
          AND (%s::int IS NULL OR d."id" <> %s::int)
          AND (
            upper(COALESCE(d."estado"::text, '')) = 'PROCESADO'
            OR COALESCE(d."procesado", false) = true
          )
        """
        row = self.query_one(
            sql,
            (
                safe_str(contenido_hash, ""),
                safe_int(exclude_documento_id, None),
                safe_int(exclude_documento_id, None),
            ),
        )
        if not row:
            return {"total": 0, "documento_ids": []}
        return {
            "total": safe_int(row.get("total"), 0) or 0,
            "documento_ids": to_json_safe(row.get("documento_ids") or []),
        }

    def mark_documento_pending_processing(
        self,
        documento_id: int,
        updated_by: Optional[int],
        error_code: str,
        error_message: str,
        mime_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Marca documento en PENDIENTE_PROCESAMIENTO con detalle de error."""
        patch = {
            "ocr_embedding_pipeline": {
                "last_error": {
                    "code": safe_str(error_code, ""),
                    "message": safe_str(error_message, ""),
                    "mime_type": normalize_mime_type(mime_type),
                    "timestamp_utc": utc_now_iso(),
                },
                "estado_documento": "PENDIENTE_PROCESAMIENTO",
            }
        }
        sql = """
        UPDATE "GestorDocumental"."Documentos"
        SET
          "estado" = 'PENDIENTE_PROCESAMIENTO',
          "metadatosExtra" = COALESCE("metadatosExtra"::jsonb, '{}'::jsonb) || %s::jsonb,
          "updatedAt" = (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
          "updatedBy" = COALESCE(%s, "updatedBy")
        WHERE "id" = %s
        RETURNING
          "id" AS documento_id,
          "archivoNombre" AS archivo_nombre,
          "estado" AS estado_documento,
          "updatedAt" AS updated_at
        """
        return self.execute_returning_one(
            sql,
            (
                json_dumps_safe(patch),
                safe_int(updated_by, None),
                int(documento_id),
            ),
        )

    def read_large_object(self, oid: int) -> bytes:
        """Reads pg_largeobject bytes from loOid."""
        try:
            lobj = self.conn.lobject(oid=int(oid), mode="rb")
            data = lobj.read()
            lobj.close()
            self.conn.commit()
            return data
        except Exception:
            self.conn.rollback()
            raise

    def ensure_queue(
        self,
        queue_name: str,
        description: str,
        max_concurrency: int,
        timeout_seconds: int,
        retries_max: int,
        priority_default: int,
    ) -> Dict[str, Any]:
        """Creates or updates queue configuration."""
        sql = """
        INSERT INTO "Operaciones"."ColasProcesamiento"
        ("nombre","descripcion","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
         "jobsPendientes","jobsProcesando","activa","createdAt")
        VALUES (%s,%s,%s,%s,%s,%s,0,0,true,(CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'))
        ON CONFLICT ("nombre") DO UPDATE
        SET "descripcion" = COALESCE(EXCLUDED."descripcion","Operaciones"."ColasProcesamiento"."descripcion"),
            "maxConcurrencia" = EXCLUDED."maxConcurrencia",
            "prioridadDefault" = EXCLUDED."prioridadDefault",
            "timeoutSegundos" = EXCLUDED."timeoutSegundos",
            "reintentosMax" = EXCLUDED."reintentosMax",
            "activa" = true
        RETURNING "id","nombre","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
                  "activa",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(
            sql,
            (
                queue_name,
                description,
                int(max_concurrency),
                int(priority_default),
                int(timeout_seconds),
                int(retries_max),
            ),
        )
        return row or {}

    def acquire_queue_slot(self, queue_name: str) -> Dict[str, Any]:
        """Acquires one queue slot if concurrency allows."""
        sql = """
        UPDATE "Operaciones"."ColasProcesamiento"
        SET "jobsProcesando" = COALESCE("jobsProcesando",0) + 1
        WHERE "nombre" = %s
          AND "activa" = true
          AND COALESCE("jobsProcesando",0) < COALESCE("maxConcurrencia",1)
        RETURNING "id","nombre","maxConcurrencia","prioridadDefault","timeoutSegundos","reintentosMax",
                  "activa",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name,))
        return {"acquired": bool(row), "queue": row}

    def release_queue_slot(self, queue_name: str) -> Dict[str, Any]:
        """Releases one queue slot."""
        sql = """
        UPDATE "Operaciones"."ColasProcesamiento"
        SET "jobsProcesando" = GREATEST(COALESCE("jobsProcesando",0) - 1, 0)
        WHERE "nombre" = %s
        RETURNING "id","nombre",COALESCE("jobsPendientes",0) AS "jobsPendientes",
                  COALESCE("jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name,))
        return {"released": bool(row), "queue": row}

    def refresh_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Refreshes queue counters from Operaciones.JobsProcesamiento."""
        sql = """
        WITH agg AS (
          SELECT
            count(*) FILTER (WHERE j."estado" = ('PENDIENTE'::"Operaciones"."EstadoJob"))::int AS pendientes,
            count(*) FILTER (WHERE j."estado" = ('EN_PROCESO'::"Operaciones"."EstadoJob"))::int AS procesando
          FROM "Operaciones"."JobsProcesamiento" j
          WHERE j."tipo" = %s
        )
        UPDATE "Operaciones"."ColasProcesamiento" c
        SET "jobsPendientes" = agg.pendientes,
            "jobsProcesando" = agg.procesando
        FROM agg
        WHERE c."nombre" = %s
        RETURNING c."id", c."nombre",
                  COALESCE(c."jobsPendientes",0) AS "jobsPendientes",
                  COALESCE(c."jobsProcesando",0) AS "jobsProcesando"
        """
        row = self.execute_returning_one(sql, (queue_name, queue_name))
        return row or {}

    def create_job(
        self,
        job_type: str,
        estado: str,
        prioridad: int,
        documento_id: Optional[int],
        parametros: Dict[str, Any],
        max_intentos: int,
    ) -> int:
        """Creates one job in Operaciones.JobsProcesamiento."""
        sql = """
        INSERT INTO "Operaciones"."JobsProcesamiento"
        ("tipo","estado","prioridad","documentoId","parametros","resultado","errorMensaje","intentos","maxIntentos",
         "programadoPara","inicio","fin","workerId","createdAt")
        VALUES (
            %s,
            CAST(%s AS "Operaciones"."EstadoJob"),
            %s,
            %s,
            %s,
            '{}'::text,
            NULL,
            0,
            %s,
            "BrainVtCommons"."getBogotaTime"(),
            CASE WHEN %s='EN_PROCESO' THEN "BrainVtCommons"."getBogotaTime"() ELSE NULL END,
            NULL,
            NULL,
            "BrainVtCommons"."getBogotaTime"()
        )
        RETURNING "id"
        """
        row = self.execute_returning_one(
            sql,
            (
                job_type,
                estado,
                int(prioridad),
                documento_id,
                json_dumps_safe(parametros),
                int(max_intentos),
                estado,
            ),
        )
        if row is None:
            raise RuntimeError("No fue posible crear job de procesamiento.")
        return int(row["id"])

    def update_job_state(
        self,
        job_id: int,
        estado: str,
        resultado: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        set_inicio_if_null: bool = False,
        set_fin: bool = False,
        clear_error: bool = False,
    ) -> None:
        """Updates job status, result payload and timestamps."""
        sets: List[str] = ['"estado" = CAST(%s AS "Operaciones"."EstadoJob")']
        params: List[Any] = [estado]

        if resultado is not None:
            sets.append('"resultado" = %s')
            params.append(json_dumps_safe(resultado))
        if error_message is not None:
            sets.append('"errorMensaje" = %s')
            params.append(error_message)
        elif clear_error:
            sets.append('"errorMensaje" = NULL')

        if set_inicio_if_null:
            sets.append('"inicio" = COALESCE("inicio","BrainVtCommons"."getBogotaTime"())')
        if set_fin:
            sets.append('"fin" = "BrainVtCommons"."getBogotaTime"()')

        sql = f'UPDATE "Operaciones"."JobsProcesamiento" SET {",".join(sets)} WHERE "id" = %s'
        params.append(int(job_id))
        self.execute(sql, tuple(params))

    def log_to_operaciones(
        self,
        nivel: str,
        modulo: str,
        mensaje: str,
        contexto: Optional[str] = None,
        stack_trace: Optional[str] = None,
        also_error: bool = False,
        error_tipo: Optional[str] = None,
    ) -> None:
        """Registra en Operaciones.LogsSistema y opcionalmente en Operaciones.Errores.

        Args:
            nivel: INFO, WARNING, ERROR, CRITICAL (se castea a Operaciones.NivelLog)
            modulo: nombre del modulo (ej. 'ocr_chunking')
            mensaje: descripcion del evento
            contexto: JSON string con detalles adicionales
            stack_trace: traceback si aplica
            also_error: si True, tambien inserta en Operaciones.Errores
            error_tipo: tipo de error para Operaciones.Errores (ej. 'PIPELINE_ERROR')
        """
        try:
            self.execute(
                'INSERT INTO "Operaciones"."LogsSistema" '
                '("timestamp","nivel","modulo","mensaje","contexto","stackTrace") '
                'VALUES ("BrainVtCommons"."getBogotaTime"(),'
                'CAST(%s AS "Operaciones"."NivelLog"),%s,%s,%s,%s)',
                (nivel, modulo, mensaje[:2000], contexto, stack_trace),
            )
        except Exception:
            LOGGER.debug("log_to_operaciones: LogsSistema insert failed", exc_info=True)

        if also_error and nivel in ("ERROR", "CRITICAL"):
            try:
                self.execute(
                    'INSERT INTO "Operaciones"."Errores" '
                    '("timestamp","tipo","modulo","mensaje","contexto","stackTrace","resuelto") '
                    'VALUES ("BrainVtCommons"."getBogotaTime"(),%s,%s,%s,%s,%s,false)',
                    (error_tipo or "PIPELINE_ERROR", modulo, mensaje[:2000], contexto, stack_trace),
                )
            except Exception:
                LOGGER.debug("log_to_operaciones: Errores insert failed", exc_info=True)

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Gets one job row."""
        sql = """
        SELECT
          "id","tipo","estado","documentoId","prioridad","intentos","maxIntentos",
          "parametros","resultado","errorMensaje","programadoPara","inicio","fin","workerId","createdAt"
        FROM "Operaciones"."JobsProcesamiento"
        WHERE "id" = %s
        """
        return self.query_one(sql, (int(job_id),))

    def count_existing_embeddings(self, documento_id: Optional[int]) -> int:
        """Counts existing embeddings for one documentoId (overwrite a nivel documento)."""
        if documento_id is None:
            return 0
        row = self.query_one(
            'SELECT COUNT(*)::int AS c FROM "IaCore"."Embeddings" WHERE "documentoId" = %s',
            (int(documento_id),),
        )
        return int(row["c"]) if row else 0

    def delete_existing_embeddings(self, documento_id: Optional[int]) -> int:
        """Deletes existing embeddings for one documentoId (overwrite a nivel documento)."""
        if documento_id is None:
            return 0
        return self.execute(
            'DELETE FROM "IaCore"."Embeddings" WHERE "documentoId" = %s',
            (int(documento_id),),
        )

    def insert_embeddings(self, rows: List[Tuple[Any, ...]]) -> int:
        """Bulk inserts rows into IaCore.Embeddings."""
        if not rows:
            return 0
        sql = """
        INSERT INTO "IaCore"."Embeddings" (
            "chunkFin",
            "chunkIndex",
            "chunkInicio",
            "chunkTexto",
            "createdAt",
            "createdBy",
            "documentoId",
            "jobFileId",
            "metadata",
            "modelo",
            "tokens",
            "updatedAt",
            "vector"
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            (CURRENT_TIMESTAMP AT TIME ZONE 'America/Bogota'),
            %s::vector
        )
        """
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, sql, rows, page_size=128)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(rows)


class QueueOptions(BaseModel):
    """Queue options."""

    enabled: bool = Field(default=True, description="Enable queue controls.")
    max_concurrency: int = Field(default=2, ge=1, le=64)
    queue_when_busy: bool = Field(
        default=True,
        description="Si la cola esta ocupada, crea job PENDIENTE (ENQUEUED) en lugar de fallar.",
    )


class OverwriteOptions(BaseModel):
    """Overwrite/guard options for embeddings y reproceso documental."""

    enabled: bool = Field(
        default=False,
        description="Si true, borra embeddings existentes del documento antes de insertar.",
    )
    allow_duplicate_hash: bool = Field(
        default=False,
        description="Permite procesar documentos aunque exista otro PROCESADO con mismo contenidoHash.",
    )
    allow_reprocess_processed: bool = Field(
        default=False,
        description="Permite reprocesar documento cuando estado actual es PROCESADO.",
    )


class ExtractionOptions(BaseModel):
    """Text/OCR extraction options."""

    engine: str = Field(default="auto", description="auto | docling | pymupdf")
    enable_pymupdf_fast_path: bool = Field(default=True)
    fast_path_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    force_full_page_ocr: bool = Field(default=False)
    do_table_structure: bool = Field(default=True)
    images_scale: float = Field(default=1.5, ge=0.5, le=3.0)
    probe_max_pages: int = Field(default=DEFAULT_PROBE_MAX_PAGES, ge=1, le=5000)
    page_mode: str = Field(default="full", description="full | head_tail")
    head_pages: int = Field(default=5, ge=1, le=1000)
    tail_pages: int = Field(default=5, ge=1, le=1000)


class CleaningOptions(BaseModel):
    """Deterministic text-cleaning options."""

    enabled: bool = Field(default=True)
    remove_headers: bool = Field(default=True)
    remove_isolated_numbers: bool = Field(default=True)
    header_threshold: int = Field(default=3, ge=2, le=20)
    remove_noisy_sentences: bool = Field(
        default=True,
        description="Elimina oraciones con exceso de tokens cortos de bajo valor semantico.",
    )
    noisy_min_alpha_tokens: int = Field(default=8, ge=3, le=200)
    noisy_short_token_ratio: float = Field(default=0.70, ge=0.0, le=1.0)


class ChunkingOptions(BaseModel):
    """Chunking options."""

    strategy: str = Field(default="semantic", description="semantic | simple")
    max_chunks: int = Field(default=0, ge=0, description="0 means unlimited.")
    simple_chunk_size: int = Field(default=1500, ge=100, le=50000)
    simple_chunk_overlap: int = Field(default=200, ge=0, le=10000)
    min_text_chars: int = Field(default=50, ge=1, le=100000)
    enable_simple_fallback: bool = Field(
        default=False,
        description="Si true y semantic no genera chunks, usa simple chunking como fallback.",
    )


class EmbeddingOptions(BaseModel):
    """Embedding generation and persistence options."""

    enabled: bool = Field(default=True)
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Cantidad de chunks procesados por lote al generar embeddings.",
    )
    save_to_db: bool = Field(default=True)
    return_vectors: bool = Field(default=False)
    require_inserted_rows: bool = Field(default=True)
    created_by_default: int = Field(default=DEFAULT_CREATED_BY)


class MockOptions(BaseModel):
    """Mock mode options."""

    enabled: bool = Field(default=False, description="Enable mock pipeline.")
    fail_phase: Optional[str] = Field(default=None, description="If set, mock fails at this phase.")
    latency_ms: int = Field(default=0, ge=0, le=120000)
    without_db: bool = Field(default=True)


class OCRChunkingRequest(BaseModel):
    """Input request model."""

    oid: int = Field(
        ...,
        description=(
            "OID del Large Object (pg_largeobject) del archivo a procesar. "
            "El documentoId de GestorDocumental se resuelve internamente por metadata/nombre."
        ),
    )
    nombre_documento: Optional[str] = Field(
        default=None,
        description="Nombre del documento. Si se envia, no se intenta resolver nombre por OID.",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Alias opcional de nombre_documento para compatibilidad.",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="Mime type esperado del archivo, por ejemplo application/pdf.",
    )
    job_filde_id: Optional[int] = Field(
        default=None,
        description="Id de archivo/job para trazabilidad (compatibilidad con nombre solicitado).",
    )
    job_field_id: Optional[int] = Field(
        default=None,
        description="Alias corregido de job_filde_id para compatibilidad con versiones recientes.",
    )
    documento_id: Optional[int] = Field(
        default=None,
        description=(
            "documentoId de GestorDocumental.Documentos ya resuelto por el caller. "
            "Si se provee, omite la resolución por metadatosExtra/archivoNombre y lo usa directamente."
        ),
    )
    usuario_proceso: Optional[str] = Field(
        default=None,
        description="Usuario funcional que ejecuta la solicitud.",
    )
    job_proceso: Optional[str] = Field(
        default=None,
        description="Nombre o identificador funcional del job de negocio.",
    )
    created_by: Optional[int] = Field(default=None)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metadata del documento. Recomendado: nombre_documento, metadata_documento"
            " (ej: paginas, idioma) y ruta_pdf."
        ),
    )
    queue: QueueOptions = Field(default_factory=QueueOptions)
    overwrite: OverwriteOptions = Field(default_factory=OverwriteOptions)
    extraction: ExtractionOptions = Field(default_factory=ExtractionOptions)
    cleaning: CleaningOptions = Field(default_factory=CleaningOptions)
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    embedding: EmbeddingOptions = Field(default_factory=EmbeddingOptions)
    mock: MockOptions = Field(default_factory=MockOptions)

    model_config = {
        "json_schema_extra": {
            "example": {
                "oid": 2299268,
                "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
                "file_name": None,
                "mime_type": "application/pdf",
                "job_filde_id": 4567,
                "usuario_proceso": "analista_anh",
                "job_proceso": "JOB_OCR_20260309_001",
                "created_by": 1101,
                "metadata": {
                    "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
                    "metadata_documento": {"paginas": 133, "idioma": "es"},
                    "ruta_pdf": "\\\\servidor\\share\\CTO_EyP_LLA_50_2013.pdf",
                },
                "queue": {
                    "enabled": True,
                    "max_concurrency": 2,
                    "queue_when_busy": True,
                },
                "overwrite": {"enabled": False, "allow_duplicate_hash": False, "allow_reprocess_processed": False},
                "extraction": {
                    "engine": "auto",
                    "enable_pymupdf_fast_path": True,
                    "fast_path_confidence_threshold": 0.85,
                    "page_mode": "head_tail",
                    "head_pages": 8,
                    "tail_pages": 8,
                },
                "cleaning": {"enabled": True, "remove_headers": True, "remove_isolated_numbers": True},
                "chunking": {"strategy": "semantic", "max_chunks": 0, "enable_simple_fallback": False},
                "embedding": {
                    "enabled": True,
                    "model_name": "intfloat/multilingual-e5-large-instruct",
                    "save_to_db": True,
                    "return_vectors": False,
                },
                "mock": {"enabled": False},
            }
        }
    }


class OCRChunkingResponse(BaseModel):
    """Output response model."""

    status: str = Field(description="COMPLETED | ENQUEUED | FAILED")
    exitoso: bool
    message: str
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error estructurado: phase, code, message, details.",
    )
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class OCRChunkingBatchRequest(BaseModel):
    """Batch request model."""

    requests: List[OCRChunkingRequest] = Field(..., min_length=1, max_length=200)
    parallel_workers: int = Field(default=1, ge=1, le=16)


class OCRChunkingBatchResponse(BaseModel):
    """Batch response model."""

    status: str
    exitoso: bool
    message: str
    total: int
    completados: int
    fallidos: int
    resultados: List[OCRChunkingResponse] = Field(default_factory=list)


def load_tokenizer(model_name: str) -> Any:
    """Loads tokenizer with in-memory cache."""
    with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(model_name)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tokenizer
        return tokenizer


def load_embedding_model(model_name: str) -> Tuple[Any, Any, str]:
    """Loads tokenizer/model/device with in-memory cache.

    Uses float16 on CUDA to halve VRAM usage (sufficient for embedding inference).
    """
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(model_name, dtype=dtype)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model, device)
        return tokenizer, model, device


def _build_accelerator_options() -> AcceleratorOptions:
    """Builds AcceleratorOptions: CUDA if available, else CPU."""
    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    return AcceleratorOptions(device=device)


def load_docling_converter(force_full_page_ocr: bool, do_table_structure: bool, images_scale: float) -> Any:
    """Loads Docling converter with GPU acceleration following official docling guide.

    Uses ThreadedPdfPipelineOptions with RapidOCR torch backend for GPU OCR,
    AcceleratorOptions for layout/table models, and tuned batch sizes.
    See: https://docling-project.github.io/docling/usage/gpu/
    """
    key = f"{int(force_full_page_ocr)}|{int(do_table_structure)}|{images_scale:.3f}"
    with _DOCLING_LOCK:
        cached = _DOCLING_CONVERTER_CACHE.get(key)
        if cached is not None:
            return cached

        use_gpu = torch.cuda.is_available()

        # RapidOCR with torch backend is the only known working GPU OCR setup.
        ocr_options = RapidOcrOptions(
            backend="torch" if use_gpu else "onnxruntime",
            force_full_page_ocr=bool(force_full_page_ocr),
        )

        options = ThreadedPdfPipelineOptions(
            do_ocr=True,
            do_table_structure=bool(do_table_structure),
            images_scale=float(images_scale),
            generate_page_images=False,
            generate_picture_images=False,
            ocr_options=ocr_options,
            accelerator_options=_build_accelerator_options(),
            document_timeout=PAGE_CHUNK_DOCUMENT_TIMEOUT if PAGE_CHUNK_DOCUMENT_TIMEOUT > 0 else None,
            # Batch sizes: increase for GPU, keep low for CPU.
            ocr_batch_size=64 if use_gpu else 4,
            layout_batch_size=64 if use_gpu else 4,
            table_batch_size=4,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=options,
                ),
            },
        )
        _DOCLING_CONVERTER_CACHE[key] = converter
        return converter


def load_docling_converter_generic() -> Any:
    """Loads generic Docling converter (no formato fijo) with GPU if available."""
    key = "GENERIC_DEFAULT"
    with _DOCLING_LOCK:
        cached = _DOCLING_CONVERTER_CACHE.get(key)
        if cached is not None:
            return cached
        converter = DocumentConverter()
        _DOCLING_CONVERTER_CACHE[key] = converter
        return converter


def apply_page_selection(pdf_bytes: bytes, page_mode: str, head_pages: int, tail_pages: int) -> Tuple[bytes, Dict[str, Any]]:
    """Applies full/head_tail selection and returns selected PDF bytes."""
    mode = safe_str(page_mode, "full").strip().lower()
    if mode not in {"full", "head_tail"}:
        raise PipelineError("PAGE_SELECTION", "INVALID_PAGE_MODE", "page_mode invalido", {"page_mode": mode})
    if mode == "full":
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = int(doc.page_count)
        return pdf_bytes, {"mode": "full", "total_pages": total_pages, "selected_pages": total_pages}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = int(doc.page_count)
        head_n = max(1, int(head_pages))
        tail_n = max(1, int(tail_pages))
        front = list(range(min(head_n, total_pages)))
        back = list(range(max(total_pages - tail_n, 0), total_pages))
        selected_indexes = sorted(set(front + back))

        with fitz.open() as out_doc:
            for idx in selected_indexes:
                out_doc.insert_pdf(doc, from_page=idx, to_page=idx)
            selected_pdf = out_doc.tobytes(garbage=4, deflate=True)

    return selected_pdf, {
        "mode": "head_tail",
        "total_pages": total_pages,
        "selected_pages": len(selected_indexes),
        "head_pages": head_n,
        "tail_pages": tail_n,
        "selected_indexes_0_based": selected_indexes,
    }


def probe_pdf_extractability(pdf_bytes: bytes, max_pages: int) -> Dict[str, Any]:
    """Probes text extractability and image presence with PyMuPDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = int(doc.page_count)
        sample_n = min(total_pages, max(1, int(max_pages)))

        pages_with_text = 0
        pages_with_images = 0
        total_chars = 0
        for idx in range(sample_n):
            page = doc.load_page(idx)
            page_text = page.get_text("text") or ""
            page_chars = len(page_text.strip())
            if page_chars > 20:
                pages_with_text += 1
            if len(page.get_images(full=True)) > 0:
                pages_with_images += 1
            total_chars += page_chars

    text_page_ratio = float(pages_with_text) / float(sample_n) if sample_n > 0 else 0.0
    avg_chars_per_page = float(total_chars) / float(sample_n) if sample_n > 0 else 0.0
    density_score = min(1.0, avg_chars_per_page / 1000.0)
    image_ratio = float(pages_with_images) / float(sample_n) if sample_n > 0 else 0.0
    image_penalty = 0.15 if image_ratio > 0.5 else 0.0
    confidence = max(0.0, min(1.0, (0.6 * text_page_ratio) + (0.4 * density_score) - image_penalty))

    return {
        "total_pages": total_pages,
        "sample_pages": sample_n,
        "pages_with_text": pages_with_text,
        "pages_with_images": pages_with_images,
        "sample_total_chars": total_chars,
        "sample_avg_chars_per_page": round(avg_chars_per_page, 3),
        "text_page_ratio": round(text_page_ratio, 6),
        "image_ratio": round(image_ratio, 6),
        "extractable_confidence": round(confidence, 6),
        "is_text_extractable": confidence >= 0.85,
    }


def extract_text_pymupdf(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extracts text with PyMuPDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = int(doc.page_count)
        parts: List[str] = []
        for idx in range(page_count):
            page = doc.load_page(idx)
            parts.append(page.get_text("text") or "")
    return "\n\n".join(parts).strip(), page_count


def convertir_valor_confianza(value: Any) -> Any:
    """Convierte valores de confianza Docling a tipos serializables."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return round(f, 6)
    if hasattr(value, "name"):
        return safe_str(getattr(value, "name"), "").lower()
    return value


def extraer_confianza_docling(confidence_obj: Any) -> Dict[str, Any]:
    """Extrae campos de confianza desde objetos de Docling."""
    if confidence_obj is None:
        return {"disponible": False}
    return {
        "mean_score": convertir_valor_confianza(getattr(confidence_obj, "mean_score", None)),
        "mean_grade": convertir_valor_confianza(getattr(confidence_obj, "mean_grade", None)),
        "low_score": convertir_valor_confianza(getattr(confidence_obj, "low_score", None)),
        "low_grade": convertir_valor_confianza(getattr(confidence_obj, "low_grade", None)),
        "layout_score": convertir_valor_confianza(getattr(confidence_obj, "layout_score", None)),
        "ocr_score": convertir_valor_confianza(getattr(confidence_obj, "ocr_score", None)),
        "parse_score": convertir_valor_confianza(getattr(confidence_obj, "parse_score", None)),
        "table_score": convertir_valor_confianza(getattr(confidence_obj, "table_score", None)),
        "disponible": True,
    }


def extract_docling_confidence_bundle(result: Any) -> Dict[str, Any]:
    """Extrae confianza global/sintetica de Docling (sin detalle pagina a pagina)."""
    global_conf = extraer_confianza_docling(getattr(result, "confidence", None))
    document = getattr(result, "document", None)
    doc_pages = getattr(document, "pages", None)
    if isinstance(doc_pages, dict):
        total_pages = len(doc_pages.keys())
    elif isinstance(doc_pages, list):
        total_pages = len(doc_pages)
    else:
        total_pages = 0
    summary = {
        "pages_total": int(total_pages),
        "global_mean_score": safe_float(global_conf.get("mean_score"), None),
        "global_ocr_score": safe_float(global_conf.get("ocr_score"), None),
    }
    selected_quality = (
        summary["global_mean_score"]
        if summary["global_mean_score"] is not None
        else summary["global_ocr_score"]
    )
    summary["selected_quality"] = selected_quality

    return {
        "global": global_conf,
        "summary": summary,
        "disponible": safe_bool(global_conf.get("disponible"), False),
    }


def _estimate_pdf_weight(pdf_bytes: bytes, sample_pages: int = 5) -> str:
    """Estima si un PDF es 'heavy' (rico en imagenes) o 'light' (rico en texto).

    Muestrea las primeras N paginas. Si >60% tienen imagenes o el promedio
    de bytes/pagina > 100KB, retorna 'heavy'. Caso contrario 'light'.
    Se usa para elegir el tamano de chunk: heavy=25 paginas, light=50 paginas.
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total = doc.page_count
            sample = min(sample_pages, total)
            pages_with_images = 0
            total_bytes = len(pdf_bytes)
            avg_bytes_per_page = total_bytes / max(total, 1)

            for i in range(sample):
                page = doc[i]
                if page.get_images():
                    pages_with_images += 1

            image_ratio = pages_with_images / max(sample, 1)
            if image_ratio > 0.6 or avg_bytes_per_page > 100_000:
                return "heavy"
            return "light"
    except Exception:
        return "heavy"  # Default to safe chunk size


def _adaptive_chunk_size(pdf_bytes: bytes) -> int:
    """Retorna el tamano optimo de chunk segun el peso del contenido del PDF."""
    weight = _estimate_pdf_weight(pdf_bytes)
    if weight == "heavy":
        return PAGE_CHUNK_SIZE_HEAVY
    return PAGE_CHUNK_SIZE


def _split_pdf_into_chunks(pdf_bytes: bytes, chunk_size: int) -> List[Tuple[bytes, int, int]]:
    """Splits a PDF into page-range chunks using PyMuPDF.

    Returns list of (chunk_bytes, start_page_0based, end_page_0based).
    Each chunk has at most `chunk_size` pages.
    """
    chunks: List[Tuple[bytes, int, int]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total) - 1
            with fitz.open() as out:
                out.insert_pdf(doc, from_page=start, to_page=end)
                chunks.append((out.tobytes(garbage=4, deflate=True), start, end))
    return chunks


def _is_ole2_binary(data: bytes) -> bool:
    """Detecta si el binario es formato OLE2 (legacy .doc/.xls/.ppt)."""
    return len(data) >= 4 and data[:4] == b"\xd0\xcf\x11\xe0"


def _convert_legacy_office(binary_bytes: bytes, source_ext: str, target_format: str) -> Tuple[bytes, str]:
    """Convierte archivo legacy Office (OLE2) a formato Open XML usando LibreOffice.

    Args:
        binary_bytes: contenido binario del archivo legacy
        source_ext: extensión original (.doc, .xls, .ppt)
        target_format: formato destino para LibreOffice (docx, xlsx, pptx)

    Returns:
        (bytes convertidos, extensión resultante)
    """
    import subprocess
    import shutil

    libreoffice_cmd = shutil.which("libreoffice") or shutil.which("soffice")
    if not libreoffice_cmd:
        # Buscar en rutas estandar de instalacion (Windows y Linux)
        for candidate in [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/usr/local/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/snap/bin/libreoffice",
        ]:
            if os.path.isfile(candidate):
                libreoffice_cmd = candidate
                break
    if not libreoffice_cmd:
        raise PipelineError(
            "LEGACY_CONVERSION", "LIBREOFFICE_NOT_FOUND",
            "LibreOffice no está instalado. Se requiere para convertir archivos "
            f"legacy ({source_ext}). Instale LibreOffice o convierta el archivo "
            f"a {target_format} manualmente.",
            {"source_ext": source_ext, "target_format": target_format},
            retryable=False,
        )

    tmpdir = tempfile.mkdtemp(prefix="lo_convert_")
    src_path = os.path.join(tmpdir, f"input{source_ext}")
    try:
        with open(src_path, "wb") as f:
            f.write(binary_bytes)

        cmd = [
            libreoffice_cmd, "--headless", "--norestore",
            "--convert-to", target_format,
            "--outdir", tmpdir,
            src_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if proc.returncode != 0:
            raise PipelineError(
                "LEGACY_CONVERSION", "LIBREOFFICE_FAILED",
                f"LibreOffice falló al convertir {source_ext} → {target_format}.",
                {"returncode": proc.returncode, "stderr": proc.stderr[:500]},
            )

        target_ext = f".{target_format}"
        converted_path = os.path.join(tmpdir, f"input{target_ext}")
        if not os.path.exists(converted_path):
            # Buscar cualquier archivo convertido
            for fname in os.listdir(tmpdir):
                if fname.endswith(target_ext):
                    converted_path = os.path.join(tmpdir, fname)
                    break

        if not os.path.exists(converted_path):
            raise PipelineError(
                "LEGACY_CONVERSION", "OUTPUT_NOT_FOUND",
                f"LibreOffice no generó el archivo convertido.",
                {"tmpdir_contents": os.listdir(tmpdir)},
            )

        with open(converted_path, "rb") as f:
            converted_bytes = f.read()

        LOGGER.info(
            "Legacy conversion: %s → %s (%d → %d bytes)",
            source_ext, target_format, len(binary_bytes), len(converted_bytes),
        )
        return converted_bytes, target_ext

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _maybe_convert_legacy_office(binary_bytes: bytes, file_name: str, mime_type: str) -> Tuple[bytes, str, str]:
    """Si el binario es OLE2 (legacy Office), lo convierte a OOXML.

    Returns:
        (bytes_a_usar, suffix_a_usar, mime_final)
    """
    if not _is_ole2_binary(binary_bytes):
        return binary_bytes, MIME_TO_EXTENSION.get(mime_type, ".bin"), mime_type

    ext = os.path.splitext(normalize_file_name(file_name))[1].lower() if file_name else ""
    conversion_map = {
        ".doc": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ".xls": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ".ppt": ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    }

    if ext not in conversion_map:
        # OLE2 sin extensión conocida → intentar como docx
        ext = ".doc"

    target_format, target_mime = conversion_map[ext]
    LOGGER.info("Detectado formato legacy OLE2 (%s). Convirtiendo a %s...", ext, target_format)

    converted_bytes, target_ext = _convert_legacy_office(binary_bytes, ext, target_format)
    return converted_bytes, target_ext, target_mime


def _convert_single_block(converter: Any, block_bytes: bytes, suffix: str) -> Tuple[str, int, Dict[str, Any]]:
    """Runs docling converter on a single PDF block and returns (text, pages, meta)."""
    temp = tempfile.NamedTemporaryFile(prefix="docling_blk_", suffix=suffix, delete=False)
    temp_path = temp.name
    temp.write(block_bytes)
    temp.flush()
    temp.close()
    try:
        result = converter.convert(temp_path)
        status = getattr(result, "status", None)
        status_str = safe_str(status, "")
        success = (
            status == ConversionStatus.SUCCESS
            or status_str.endswith("SUCCESS")
            or status_str.endswith("PARTIAL_SUCCESS")
        )
        if not success:
            raise PipelineError(
                "TEXT_EXTRACTION", "DOCLING_CONVERSION_FAILED",
                "Docling no pudo convertir el bloque.",
                {"status": status_str},
            )
        document = getattr(result, "document", None)
        if document is None:
            return "", 0, {"conversion_status": status_str}

        text = safe_str(getattr(document, "export_to_markdown", lambda: "")(), "")
        if not text.strip():
            text = safe_str(getattr(document, "export_to_text", lambda: "")(), "")
        page_count = int(len(getattr(document, "pages", []) or []))
        confidence_bundle = extract_docling_confidence_bundle(result)
        return text.strip(), page_count, {
            "conversion_status": status_str,
            "docling_confidence": confidence_bundle,
            "ocr_quality": safe_float(
                (confidence_bundle.get("summary") or {}).get("selected_quality"), None,
            ),
        }
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def extract_text_docling(binary_bytes: bytes, extraction: ExtractionOptions, mime_type: str, file_name: str) -> Tuple[str, int, Dict[str, Any]]:
    """Extracts text/OCR with Docling for supported mime types.

    For large PDFs (> PAGE_CHUNK_SIZE pages), the document is split into
    page-range blocks processed independently. This prevents std::bad_alloc
    in docling-parse and allows garbage collection between blocks.

    For legacy Office formats (OLE2: .doc/.xls/.ppt), converts to OOXML
    using LibreOffice before passing to Docling.
    """
    # Convertir formatos legacy OLE2 si es necesario
    binary_bytes, converted_suffix, converted_mime = _maybe_convert_legacy_office(
        binary_bytes, file_name, mime_type,
    )
    normalized_mime = normalize_mime_type(converted_mime)
    is_pdf = normalized_mime == "application/pdf"
    if is_pdf:
        converter = load_docling_converter(
            force_full_page_ocr=bool(extraction.force_full_page_ocr),
            do_table_structure=bool(extraction.do_table_structure),
            images_scale=float(extraction.images_scale),
        )
    else:
        converter = load_docling_converter_generic()

    suffix = converted_suffix if converted_suffix else MIME_TO_EXTENSION.get(normalized_mime)
    if not suffix:
        _, ext = os.path.splitext(normalize_file_name(file_name))
        suffix = ext if ext else ".bin"

    # --- Chunked processing for large PDFs ---
    if is_pdf and PAGE_CHUNK_SIZE > 0:
        with fitz.open(stream=binary_bytes, filetype="pdf") as doc:
            total_pages = doc.page_count

        adaptive_size = _adaptive_chunk_size(binary_bytes)
        if total_pages > adaptive_size:
            LOGGER.info(
                "Large PDF detected (%d pages, weight=%s). Splitting into chunks of %d pages.",
                total_pages, _estimate_pdf_weight(binary_bytes), adaptive_size,
            )
            chunks = _split_pdf_into_chunks(binary_bytes, adaptive_size)
            all_texts: List[str] = []
            total_extracted_pages = 0
            last_meta: Dict[str, Any] = {}

            for i, (chunk_bytes, start_pg, end_pg) in enumerate(chunks):
                LOGGER.info(
                    "Processing chunk %d/%d (pages %d-%d)...",
                    i + 1, len(chunks), start_pg + 1, end_pg + 1,
                )
                try:
                    text, pages, meta = _convert_single_block(converter, chunk_bytes, suffix)
                    if text:
                        all_texts.append(text)
                    total_extracted_pages += pages
                    last_meta = meta
                except PipelineError:
                    LOGGER.warning(
                        "Chunk %d/%d failed (pages %d-%d), skipping.",
                        i + 1, len(chunks), start_pg + 1, end_pg + 1,
                    )
                # Allow GC to reclaim docling-parse C++ memory between chunks.
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            combined_text = "\n\n".join(all_texts).strip()
            if not combined_text:
                raise PipelineError(
                    "TEXT_EXTRACTION", "DOCLING_ALL_CHUNKS_EMPTY",
                    "Ningun bloque produjo texto.",
                )
            last_meta.update({
                "conversion_mode": "docling_ocr_chunked",
                "mime_type": normalized_mime,
                "total_pages_original": total_pages,
                "chunks_processed": len(chunks),
                "chunk_size": adaptive_size,
                "pdf_weight": _estimate_pdf_weight(binary_bytes),
            })
            return combined_text, total_extracted_pages, last_meta

    # --- Standard single-pass processing ---
    text, pages, meta = _convert_single_block(converter, binary_bytes, suffix)
    if not text:
        raise PipelineError("TEXT_EXTRACTION", "DOCLING_RESULT_EMPTY", "Docling no devolvio texto.")
    meta.update({
        "conversion_mode": "docling_ocr" if is_pdf else "docling_extract",
        "mime_type": normalized_mime,
    })
    return text, pages, meta


def sentence_is_noisy(sentence: str, min_alpha_tokens: int, short_ratio: float) -> bool:
    """Detecta oraciones OCR ruidosas (muchos tokens alfabeticos muy cortos)."""
    tokens = WORD_TOKEN_PATTERN.findall(safe_str(sentence, ""))
    if len(tokens) < int(min_alpha_tokens):
        return False
    alpha_tokens = [tok for tok in tokens if tok]
    if not alpha_tokens:
        return False
    short_count = sum(1 for tok in alpha_tokens if len(tok) <= 2)
    long_count = sum(1 for tok in alpha_tokens if len(tok) >= 4)
    ratio = float(short_count) / float(len(alpha_tokens))
    return ratio >= float(short_ratio) and long_count <= max(2, int(len(alpha_tokens) * 0.25))


def clean_noisy_sentences_in_line(line: str, min_alpha_tokens: int, short_ratio: float) -> Tuple[str, int]:
    """Limpia segmentos ruidosos por linea conservando frases legibles."""
    raw = safe_str(line, "")
    if not raw.strip():
        return raw, 0
    segments = re.split(r"(?<=[\.\!\?\;\:])\s+", raw)
    if not segments:
        return raw, 0

    kept: List[str] = []
    removed = 0
    for segment in segments:
        piece = segment.strip()
        if not piece:
            continue
        if sentence_is_noisy(piece, min_alpha_tokens=min_alpha_tokens, short_ratio=short_ratio):
            removed += 1
            continue
        kept.append(piece)

    if kept:
        return " ".join(kept), removed
    if sentence_is_noisy(raw, min_alpha_tokens=min_alpha_tokens, short_ratio=short_ratio):
        return "", max(1, removed)
    return raw, removed


def clean_text(text: str, cleaning: CleaningOptions) -> Tuple[str, Dict[str, Any]]:
    """Applies deterministic cleanup controlled by flags."""
    original = safe_str(text, "")
    if not cleaning.enabled:
        return original, {"enabled": False, "chars_before": len(original), "chars_after": len(original)}

    cleaned = PAGE_INDICATOR_PATTERN.sub("", original)
    lines = cleaned.split("\n")

    if cleaning.remove_headers:
        counts: Dict[str, int] = {}
        for line in lines:
            normalized = line.strip()
            if 5 < len(normalized) < 150:
                counts[normalized] = counts.get(normalized, 0) + 1
        headers = {line for line, count in counts.items() if count >= int(cleaning.header_threshold)}
        seen: set[str] = set()
        filtered: List[str] = []
        for line in lines:
            normalized = line.strip()
            if normalized in headers:
                if normalized in seen:
                    continue
                seen.add(normalized)
            filtered.append(line)
        lines = filtered

    if cleaning.remove_isolated_numbers:
        lines = [line for line in lines if not ISOLATED_NUMBER_PATTERN.match(line.strip()) or not line.strip()]

    removed_noisy_sentences = 0
    if cleaning.remove_noisy_sentences:
        filtered_lines: List[str] = []
        for line in lines:
            cleaned_line, removed_count = clean_noisy_sentences_in_line(
                line=line,
                min_alpha_tokens=int(cleaning.noisy_min_alpha_tokens),
                short_ratio=float(cleaning.noisy_short_token_ratio),
            )
            removed_noisy_sentences += int(removed_count)
            if cleaned_line.strip() or not line.strip():
                filtered_lines.append(cleaned_line)
        lines = filtered_lines

    merged = "\n".join(lines)
    merged = MULTI_EMPTY_LINES_PATTERN.sub("\n\n", merged).strip()
    return merged, {
        "enabled": True,
        "chars_before": len(original),
        "chars_after": len(merged),
        "remove_headers": bool(cleaning.remove_headers),
        "remove_isolated_numbers": bool(cleaning.remove_isolated_numbers),
        "remove_noisy_sentences": bool(cleaning.remove_noisy_sentences),
        "removed_noisy_sentences": int(removed_noisy_sentences),
    }


def count_words(text: str) -> int:
    """Cuenta palabras de forma simple y determinista."""
    return len(re.findall(r"\S+", safe_str(text, "")))


def build_docling_document(text: str) -> Any:
    """Builds DoclingDocument from plain text using compatible methods."""
    try:
        doc = DoclingDocument(name="documento")
        if hasattr(doc, "add_text"):
            doc.add_text(text)
            return doc
    except Exception:
        pass
    try:
        doc = DoclingDocument(name="documento")
        if hasattr(doc, "add_text"):
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
            return doc
    except Exception:
        pass
    if hasattr(DoclingDocument, "from_markdown"):
        return DoclingDocument.from_markdown(text)
    if hasattr(DoclingDocument, "from_text"):
        return DoclingDocument.from_text(text)
    raise RuntimeError("No fue posible construir DoclingDocument.")


def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking."""
    content = safe_str(text, "")
    if not content.strip():
        return []
    size = max(1, int(chunk_size))
    ov = max(0, int(overlap))
    if ov >= size:
        ov = max(0, size - 1)
    chunks: List[str] = []
    pos = 0
    total = len(content)
    while pos < total:
        end = min(pos + size, total)
        piece = content[pos:end].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        pos = end - ov
    return chunks


def semantic_chunk_text(text: str, tokenizer: Any, model_name: str, max_tokens: int) -> List[str]:
    """Semantic chunking with Docling HybridChunker using explicit max_tokens."""
    document = build_docling_document(text)
    effective_max_tokens = max(128, int(max_tokens))
    attempts: List[Tuple[str, Any]] = []
    if safe_str(model_name, "").strip():
        attempts.append(("model_name", safe_str(model_name, "").strip()))
    if tokenizer is not None:
        attempts.append(("tokenizer_object", tokenizer))
    if not attempts:
        attempts.append(("default_model", DEFAULT_EMBEDDING_MODEL))

    raw_chunks: Optional[List[Any]] = None
    errors: List[str] = []
    for source_name, token_source in attempts:
        try:
            chunker = HybridChunker(tokenizer=token_source, max_tokens=effective_max_tokens)
            raw_chunks = list(chunker.chunk(document))
            break
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {str(exc)}")

    if raw_chunks is None:
        raise RuntimeError(
            "HybridChunker fallo con tokenizer/modelo. "
            + " | ".join(errors)
            + f" | max_tokens={effective_max_tokens}"
        )

    chunks: List[str] = []
    for chunk in raw_chunks:
        if hasattr(chunk, "text"):
            value = safe_str(getattr(chunk, "text"), "").strip()
        elif isinstance(chunk, dict):
            value = safe_str(chunk.get("text"), "").strip()
        else:
            value = safe_str(chunk, "").strip()
        if value:
            chunks.append(value)
    return chunks


def rebalance_chunks(chunks: List[str], target: Optional[int]) -> List[str]:
    """Rebalances chunk list to target count if needed."""
    if target is None:
        return chunks
    wanted = max(1, int(target))
    clean = [safe_str(x, "").strip() for x in chunks if safe_str(x, "").strip()]
    if len(clean) == wanted:
        return clean
    if len(clean) > wanted:
        return clean[:wanted]
    while len(clean) < wanted:
        longest_idx = max(range(len(clean)), key=lambda idx: len(clean[idx]))
        source = clean[longest_idx]
        if len(source) < 2:
            break
        split_at = max(1, len(source) // 2)
        left = source[:split_at].strip()
        right = source[split_at:].strip()
        if not left or not right:
            break
        clean = clean[:longest_idx] + [left, right] + clean[longest_idx + 1 :]
    return clean[:wanted]


def estimate_bounds(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    """Estimates chunk start/end offsets on source text."""
    text = safe_str(full_text, "")
    cursor = 0
    text_len = len(text)
    bounds: List[Tuple[int, int]] = []
    for chunk in chunks:
        if not chunk:
            bounds.append((cursor, cursor))
            continue
        start = text.find(chunk, cursor)
        if start < 0:
            start = text.find(chunk)
        if start < 0:
            start = min(cursor, text_len)
        end = min(text_len, start + len(chunk))
        cursor = max(cursor, end)
        bounds.append((int(start), int(end)))
    return bounds


def embed_chunks(
    chunks: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    batch_size: int,
) -> Tuple[List[List[float]], List[int]]:
    """Generates normalized embeddings in batches."""
    vectors: List[List[float]] = []
    tokens_per_chunk: List[int] = []
    step = max(1, int(batch_size))

    use_amp = device == "cuda"
    for idx in range(0, len(chunks), step):
        batch_text = chunks[idx : idx + step]
        encoded = tokenizer(
            batch_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max(16, int(max_length)),
        )
        attention_mask = encoded["attention_mask"]
        if device == "cuda":
            encoded = {k: v.to(device) for k, v in encoded.items()}
            attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model_output = model(**encoded)
            else:
                model_output = model(**encoded)
        pooled = mean_pooling(model_output, attention_mask)
        normalized = torch_functional.normalize(pooled.float(), p=2, dim=1)

        vectors.extend(normalized.cpu().tolist())
        token_counts = attention_mask.cpu().sum(dim=1).tolist()
        tokens_per_chunk.extend([int(x) for x in token_counts])
    return vectors, tokens_per_chunk


def resolve_extraction_engine(extraction: ExtractionOptions, probe: Dict[str, Any]) -> str:
    """Decides extraction engine based on request and probe confidence."""
    requested = safe_str(extraction.engine, "auto").strip().lower()
    if requested not in {"auto", "docling", "pymupdf"}:
        raise PipelineError("TEXT_EXTRACTION", "INVALID_ENGINE", "extraction.engine invalido", {"engine": requested})
    if extraction.force_full_page_ocr:
        return "docling"
    if requested in {"docling", "pymupdf"}:
        return requested
    if not extraction.enable_pymupdf_fast_path:
        return "docling"

    confidence = float(probe.get("extractable_confidence", 0.0) or 0.0)
    threshold = float(extraction.fast_path_confidence_threshold)
    return "pymupdf" if confidence >= threshold else "docling"


def gpu_metrics() -> Dict[str, Any]:
    """Returns CUDA resource indicators."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0}
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_mem = int(props.total_memory)
    allocated = int(torch.cuda.memory_allocated(device_index))
    reserved = int(torch.cuda.memory_reserved(device_index))
    free_estimate = max(total_mem - reserved, 0)
    return {
        "cuda_available": True,
        "device_count": int(torch.cuda.device_count()),
        "device_index": int(device_index),
        "device_name": props.name,
        "total_memory_bytes": total_mem,
        "allocated_memory_bytes": allocated,
        "reserved_memory_bytes": reserved,
        "free_estimate_bytes": free_estimate,
    }


def derive_embedding_max_length(chunking: ChunkingOptions) -> int:
    """Deriva max_length para embeddings a partir de la estrategia de chunking."""
    approx_from_chars = max(128, int(chunking.simple_chunk_size / 4))
    return min(2048, approx_from_chars)


def build_job_payload(
    request: OCRChunkingRequest,
    file_name: str,
    mime_type: str,
    item_info: Optional[Dict[str, Any]],
    documento_info: Optional[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    """Builds compact payload for Operaciones.JobsProcesamiento."""
    documento_id_real = safe_int((documento_info or {}).get("documento_id"), None)
    return {
        "pipeline": "OCR_CHUNKING_SERVICE",
        "stage": stage,
        "oid": int(request.oid),
        "oid_documento": int(request.oid),
        "documento_id_real": documento_id_real,
        "file_name": file_name,
        "mime_type": mime_type,
        "usuario_proceso": request.usuario_proceso,
        "job_proceso": request.job_proceso,
        "job_filde_id": request.job_field_id or request.job_filde_id,
        "queue_name": DEFAULT_QUEUE_NAME,
        "overwrite_enabled": request.overwrite.enabled,
        "allow_duplicate_hash": request.overwrite.allow_duplicate_hash,
        "allow_reprocess_processed": request.overwrite.allow_reprocess_processed,
        "metadata": request.metadata,
        "resolved_item": to_json_safe(item_info or {}),
        "resolved_documento": to_json_safe(documento_info or {}),
        "created_at_utc": utc_now_iso(),
    }


def update_job_progress(
    db: PostgresClient,
    job_id: int,
    recorder: PhaseRecorder,
    pipeline_status: str,
    current_phase: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Stores progressive status in job.resultado."""
    payload = {
        "pipeline_status": pipeline_status,
        "current_phase": current_phase,
        "phases": recorder.as_list(),
        "extra": extra or {},
        "updated_at_utc": utc_now_iso(),
    }
    db.update_job_state(
        job_id=job_id,
        estado="EN_PROCESO",
        resultado=payload,
        set_inicio_if_null=True,
    )


def run_mock_pipeline(request: OCRChunkingRequest) -> OCRChunkingResponse:
    """Runs deterministic mock pipeline (no DB usage)."""
    recorder = PhaseRecorder()
    started = time.monotonic()
    latency = max(0, int(request.mock.latency_ms))
    fail_phase = safe_str(request.mock.fail_phase, "").strip().upper() if request.mock.fail_phase else None

    def phase(name: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        recorder.push(name, "OK", message, details or {})
        if latency > 0:
            time.sleep(float(latency) / 1000.0)
        if fail_phase and fail_phase == name:
            raise PipelineError(name, "MOCK_FORCED_FAILURE", f"Falla mock forzada en fase {name}.")

    try:
        phase("REQUEST_VALIDATION", "Mock request validada.")
        phase("QUEUE", "Mock cola aceptada.", {"simulated": True})
        phase("LOAD_BINARY", "Mock bytes cargados.", {"bytes": 1024})
        phase("PAGE_SELECTION", "Mock seleccion de paginas.", {"mode": request.extraction.page_mode})
        phase("PROBE", "Mock probe realizado.", {"extractable_confidence": 0.93, "is_text_extractable": True})
        phase("TEXT_EXTRACTION", "Mock extraccion completada.", {"engine": "pymupdf"})
        phase("TEXT_CLEANING", "Mock limpieza completada.", {"chars_after": 5000})
        phase("CHUNKING", "Mock chunking completado.", {"chunks": 8})
        phase("EMBEDDINGS", "Mock embeddings completados.", {"vectors": 8})
        phase("PERSIST", "Mock persistencia completada.", {"inserted_rows": 8})

        elapsed_ms = int((time.monotonic() - started) * 1000)
        return OCRChunkingResponse(
            status="COMPLETED",
            exitoso=True,
            message="Mock pipeline completado.",
            phases=recorder.as_list(),
            data={
                "oid": int(request.oid),
                "job_id": None,
                "engine_used": "pymupdf",
                "chunks_count": 8,
                "inserted_rows": 8,
                "elapsed_ms": elapsed_ms,
                "mock_mode": True,
            },
        )
    except PipelineError as exc:
        recorder.push(exc.phase, "ERROR", exc.message, exc.details)
        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Mock pipeline fallido.",
            error=exc.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "mock_mode": True},
        )


def run_real_pipeline(request: OCRChunkingRequest, stage: str = "pipeline") -> OCRChunkingResponse:
    """Runs selected stage (ocr/chunking/embedding/pipeline) against DB and model runtime."""
    stage_key = safe_str(stage, "pipeline").strip().lower()
    if stage_key not in {"ocr", "chunking", "embedding", "pipeline"}:
        raise PipelineError(
            "REQUEST_VALIDATION",
            "INVALID_STAGE",
            "Etapa invalida. Usa: ocr, chunking, embedding o pipeline.",
            {"stage": stage},
        )

    run_chunking_stage = stage_key in {"chunking", "embedding", "pipeline"}
    run_embedding_stage = stage_key in {"embedding", "pipeline"} and request.embedding.enabled
    run_persist_stage = stage_key in {"embedding", "pipeline"} and request.embedding.save_to_db

    started = time.monotonic()
    recorder = PhaseRecorder()
    queue_slot_acquired = False
    queue_name = DEFAULT_QUEUE_NAME
    job_type = DEFAULT_JOB_TYPE if stage_key == "pipeline" else f"{DEFAULT_JOB_TYPE}_{stage_key.upper()}"
    service_endpoint = SERVICE_STAGE_ENDPOINTS.get(stage_key, "/PipelineOCR/process")
    job_id: Optional[int] = None
    item_info: Optional[Dict[str, Any]] = None
    documento_info: Optional[Dict[str, Any]] = None
    oid_stats: Optional[Dict[str, Any]] = None
    file_name = normalize_file_name(request.nombre_documento or request.file_name)
    mime_type = normalize_mime_type(request.mime_type)
    mime_support_info: Dict[str, Any] = {}
    documento_id: Optional[int] = None

    recorder.push("REQUEST_VALIDATION", "OK", "Request recibida.", {"oid": int(request.oid), "stage": stage_key})

    if run_persist_stage and not request.embedding.enabled:
        err = PipelineError(
            "REQUEST_VALIDATION",
            "INVALID_EMBEDDING_CONFIG",
            "embedding.save_to_db requiere embedding.enabled=true.",
        )
        recorder.push(err.phase, "ERROR", err.message, err.details)
        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Configuracion invalida.",
            error=err.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key},
        )

    def _create_job_guarded(
        db: PostgresClient,
        estado: str,
        prioridad: int,
        parametros: Dict[str, Any],
        max_intentos: int,
    ) -> int:
        """Crea job; si FK falla (caller transaction no commiteada), reintenta con documentoId=NULL."""
        try:
            return db.create_job(
                job_type=job_type,
                estado=estado,
                prioridad=prioridad,
                documento_id=documento_id,
                parametros=parametros,
                max_intentos=max_intentos,
            )
        except psycopg2.errors.ForeignKeyViolation:
            LOGGER.warning(
                "JOB_CREATE FK fallback: documentoId=%s no visible (posible tx no commiteada del caller). "
                "Creando job con documentoId=NULL; documento_id_real queda en parametros.",
                documento_id,
            )
            db.conn.rollback()
            parametros_with_doc = dict(parametros)
            parametros_with_doc["documento_id_real"] = documento_id
            parametros_with_doc["documento_id_fk_fallback"] = True
            recorder.push(
                "JOB_CREATE",
                "WARN",
                (
                    f"FK fallback: documentoId={documento_id} no visible en GestorDocumental.Documentos "
                    "(posible transacción del caller no commiteada). "
                    "Job creado con documentoId=NULL; documento_id_real preservado en parametros."
                ),
                {
                    "oid": int(request.oid),
                    "documento_id_intended": documento_id,
                    "fallback": True,
                },
            )
            return db.create_job(
                job_type=job_type,
                estado=estado,
                prioridad=prioridad,
                documento_id=None,
                parametros=parametros_with_doc,
                max_intentos=max_intentos,
            )

    try:
        with PostgresClient(PostgresSettings.from_env()) as db:
            oid_stats = db.fetch_large_object_stats(int(request.oid))
            if oid_stats:
                recorder.push(
                    "LOAD_OID_INFO",
                    "OK",
                    "Resumen de pg_largeobject obtenido.",
                    {
                        "oid": safe_int(oid_stats.get("oid"), None),
                        "paginas_aprox": safe_int(oid_stats.get("paginas"), None),
                        "bytes_aprox": safe_int(oid_stats.get("bytes_aprox"), None),
                    },
                )
            else:
                recorder.push(
                    "LOAD_OID_INFO",
                    "WARN",
                    "No se encontraron metadatos de pg_largeobject para el OID.",
                    {"oid": int(request.oid)},
                )

            if file_name:
                recorder.push(
                    "LOAD_ITEM",
                    "SKIPPED",
                    "Nombre de documento recibido en request; no se consulta nombre por OID.",
                    {"file_name": file_name},
                )
            else:
                item_info = db.fetch_item_by_oid(int(request.oid))
                if item_info is not None:
                    file_name = normalize_file_name(item_info.get("nombre_archivo"))
                    recorder.push(
                        "LOAD_ITEM",
                        "OK",
                        "Item resuelto por OID.",
                        {
                            "item_id": safe_int(item_info.get("item_id"), None),
                            "file_name": file_name,
                            "estado_item": safe_str(item_info.get("estado"), ""),
                        },
                    )
                if not file_name:
                    file_name = f"documento_{int(request.oid)}.bin"
                    recorder.push(
                        "LOAD_ITEM",
                        "WARN",
                        "No se encontro nombre por OID; se usa nombre por defecto.",
                        {"file_name": file_name},
                    )

            # Resuelve documentoId real (FK): prioriza el valor provisto por el caller,
            # luego busca por OID en metadatosExtra y por nombre de archivo como fallback.
            caller_documento_id = safe_int(request.documento_id, None)
            if caller_documento_id is not None:
                documento_id = caller_documento_id
                documento_info = {"documento_id": caller_documento_id}
                recorder.push(
                    "LOAD_DOCUMENT",
                    "OK",
                    "documentoId recibido directamente del caller; omitiendo resolución por metadatosExtra/archivoNombre.",
                    {"documento_id": documento_id, "oid": int(request.oid)},
                )
            else:
                documento_info = db.fetch_documento_by_metadata_oid(int(request.oid))
                if documento_info is not None:
                    documento_id = safe_int(documento_info.get("documento_id"), None)
                    recorder.push(
                        "LOAD_DOCUMENT",
                        "OK",
                        "Documento resuelto por metadatosExtra.ocr.metadata.oid.",
                        {
                            "documento_id": documento_id,
                            "archivo_nombre": safe_str(documento_info.get("archivo_nombre"), ""),
                        },
                    )
                elif file_name:
                    documento_info = db.fetch_documento_by_file_name(file_name)
                    if documento_info is not None:
                        documento_id = safe_int(documento_info.get("documento_id"), None)
                        recorder.push(
                            "LOAD_DOCUMENT",
                            "OK",
                            "Documento resuelto por archivoNombre (fallback).",
                            {
                                "documento_id": documento_id,
                                "archivo_nombre": safe_str(documento_info.get("archivo_nombre"), ""),
                                "file_name_input": file_name,
                            },
                        )
                    else:
                        recorder.push(
                            "LOAD_DOCUMENT",
                            "WARN",
                            "No se resolvio documentoId en GestorDocumental.Documentos; se usara null en FK.",
                            {"oid": int(request.oid), "file_name": file_name},
                        )
                else:
                    recorder.push(
                        "LOAD_DOCUMENT",
                        "WARN",
                        "No se resolvio documentoId en GestorDocumental.Documentos; se usara null en FK.",
                        {"oid": int(request.oid)},
                    )

            # Reproceso de documento en estado PROCESADO.
            if documento_id is not None:
                estado_actual = safe_str((documento_info or {}).get("estado_documento"), "").strip().upper()
                if estado_actual == "PROCESADO" and not request.overwrite.allow_reprocess_processed:
                    raise PipelineError(
                        "DOCUMENT_GUARD",
                        "DOCUMENT_ALREADY_PROCESSED",
                        "El documento ya esta en estado PROCESADO y allow_reprocess_processed=false.",
                        {
                            "documento_id": documento_id,
                            "estado_documento": estado_actual,
                            "allow_reprocess_processed": bool(request.overwrite.allow_reprocess_processed),
                        },
                    )

            # Resolución MIME en dos pasos: primero por nombre/metadata,
            # luego por contenido binario si no se resuelve o no está soportado.
            mime_type = resolve_request_mime_type(
                request_mime_type=request.mime_type,
                request_metadata=request.metadata,
                documento_info=documento_info,
                file_name=file_name,
            )
            # Flag para re-validar después de cargar binario si MIME no se resolvió
            _mime_needs_binary_fallback = not mime_type or not is_supported_docling_mime(mime_type)
            mime_support_info = {
                "mime_type": mime_type,
                "supported": is_supported_docling_mime(mime_type),
                "supported_list": sorted(SUPPORTED_DOCLING_MIME_TYPES),
                "file_name": file_name,
                "detection_method": "name_metadata" if mime_type else "pending_binary",
            }
            if not mime_support_info["supported"] and not _mime_needs_binary_fallback:
                if documento_id is not None:
                    updated_by_guard = (
                        request.created_by
                        if request.created_by is not None
                        else request.embedding.created_by_default
                    )
                    db.mark_documento_pending_processing(
                        documento_id=documento_id,
                        updated_by=updated_by_guard,
                        error_code="UNSUPPORTED_MIME_TYPE",
                        error_message=f"Mime type no soportado: {mime_type}. Archivo: {file_name}",
                        mime_type=mime_type,
                    )
                raise PipelineError(
                    "MIME_VALIDATION",
                    "UNSUPPORTED_MIME_TYPE",
                    f"Mime type '{mime_type}' no soportado para procesamiento OCR/Docling. Archivo: {file_name}",
                    mime_support_info,
                    retryable=False,
                )
            recorder.push("MIME_VALIDATION", "OK", "Mime type validado.", mime_support_info)

            # Early-fail: si save_to_db=true y documentoId no fue resuelto, abortar antes del OCR.
            if documento_id is None and run_persist_stage and request.embedding.save_to_db:
                raise PipelineError(
                    "LOAD_DOCUMENT",
                    "MISSING_REQUIRED_DOCUMENTO_ID",
                    "documentoId no resuelto y embedding.save_to_db=true; abortando antes del OCR para evitar desperdicio de GPU.",
                    {
                        "oid": int(request.oid),
                        "file_name": file_name,
                        "hint": "Proveer documento_id en el payload o asegurarse de que el documento exista en GestorDocumental.Documentos con el OID en metadatosExtra.",
                    },
                )

            if request.queue.enabled:
                queue_info = db.ensure_queue(
                    queue_name=queue_name,
                    description=f"Servicio {stage_key} OCR/Chunking/Embeddings",
                    max_concurrency=request.queue.max_concurrency,
                    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                    retries_max=3,
                    priority_default=40,
                )
                recorder.push("QUEUE_SETUP", "OK", "Queue asegurada.", {"queue": queue_info})
            else:
                recorder.push("QUEUE_SETUP", "SKIPPED", "Queue deshabilitada.")

            payload = build_job_payload(request, file_name, mime_type, item_info, documento_info, stage_key)
            payload["service_endpoint"] = service_endpoint

            if request.queue.enabled:
                slot = db.acquire_queue_slot(queue_name)
                queue_slot_acquired = bool(slot.get("acquired"))
                if not queue_slot_acquired:
                    if request.queue.queue_when_busy:
                        job_id = _create_job_guarded(
                            db=db,
                            estado="PENDIENTE",
                            prioridad=40,
                            parametros=payload,
                            max_intentos=3,
                        )
                        db.refresh_queue_stats(queue_name)
                        recorder.push(
                            "QUEUE_ADMISSION",
                            "ENQUEUED",
                            "Cola ocupada; job en PENDIENTE.",
                            {"job_id": job_id, "queue_name": queue_name},
                        )
                        return OCRChunkingResponse(
                            status="ENQUEUED",
                            exitoso=True,
                            message="Cola ocupada, job encolado.",
                            phases=recorder.as_list(),
                            data={
                                "oid": int(request.oid),
                                "stage": stage_key,
                                "job_id": job_id,
                                "queue_name": queue_name,
                                "queue_state": slot.get("queue"),
                            },
                        )
                    raise PipelineError(
                        "QUEUE_ADMISSION",
                        "QUEUE_BUSY",
                        "No hay slots disponibles y queue_when_busy=false.",
                        {"queue_name": queue_name},
                    )

                recorder.push(
                    "QUEUE_ADMISSION",
                    "OK",
                    "Slot de cola adquirido.",
                    {"queue_name": queue_name, "queue_state": slot.get("queue")},
                )

            job_id = _create_job_guarded(
                db=db,
                estado="EN_PROCESO",
                prioridad=40,
                parametros=payload,
                max_intentos=3,
            )
            update_job_progress(db, job_id, recorder, "RUNNING", "QUEUE_ADMISSION", {"job_id": job_id})

            try:
                document_bytes = db.read_large_object(int(request.oid))
            except Exception as exc:
                raise PipelineError(
                    "LOAD_BINARY",
                    "OID_READ_FAILED",
                    "No fue posible leer el binary del OID solicitado.",
                    {"oid": int(request.oid), "error": f"{type(exc).__name__}: {str(exc)}"},
                ) from exc
            if not document_bytes:
                raise PipelineError("LOAD_BINARY", "EMPTY_BINARY", "Large object vacio.")
            binary_sha256 = hashlib.sha256(document_bytes).hexdigest()

            # Re-validar MIME con contenido binario si la detección por nombre falló
            if _mime_needs_binary_fallback:
                mime_type = resolve_request_mime_type(
                    request_mime_type=request.mime_type,
                    request_metadata=request.metadata,
                    documento_info=documento_info,
                    file_name=file_name,
                    binary_data=document_bytes,
                )
                mime_support_info = {
                    "mime_type": mime_type,
                    "supported": is_supported_docling_mime(mime_type),
                    "file_name": file_name,
                    "detection_method": "binary_fallback",
                }
                if not is_supported_docling_mime(mime_type):
                    raise PipelineError(
                        "MIME_VALIDATION",
                        "UNSUPPORTED_MIME_TYPE",
                        f"Mime type '{mime_type or 'desconocido'}' no soportado. "
                        f"Archivo: {file_name}. "
                        f"Formatos soportados: PDF, DOCX, XLSX, PPTX, HTML, Markdown, CSV, imágenes.",
                        mime_support_info,
                        retryable=False,
                    )
                LOGGER.info("MIME resuelto por contenido binario: %s (archivo: %s)", mime_type, file_name)
                recorder.push("MIME_VALIDATION", "OK", f"Mime resuelto por contenido: {mime_type}", mime_support_info)

            # Control de hash duplicado en documentos PROCESADOS.
            if not request.overwrite.allow_duplicate_hash:
                hash_guard = db.count_processed_documents_by_hash(
                    contenido_hash=binary_sha256,
                    exclude_documento_id=documento_id,
                )
                if safe_int(hash_guard.get("total"), 0) and int(hash_guard.get("total")) > 0:
                    raise PipelineError(
                        "DOCUMENT_HASH_GUARD",
                        "DUPLICATE_DOCUMENT_HASH",
                        "Ya existe un documento PROCESADO con el mismo hash y allow_duplicate_hash=false.",
                        {
                            "documento_id": documento_id,
                            "binary_sha256": binary_sha256,
                            "duplicados_total": safe_int(hash_guard.get("total"), 0),
                            "duplicados_ids": to_json_safe(hash_guard.get("documento_ids") or []),
                            "allow_duplicate_hash": bool(request.overwrite.allow_duplicate_hash),
                        },
                    )

            recorder.push(
                "LOAD_BINARY",
                "OK",
                "Binary cargado desde large object.",
                {
                    "oid": int(request.oid),
                    "bytes": len(document_bytes),
                    "sha256": binary_sha256,
                    "mime_type": mime_type,
                },
            )
            update_job_progress(db, job_id, recorder, "RUNNING", "LOAD_BINARY")
            is_pdf_document = mime_type == "application/pdf"
            selected_binary = document_bytes
            if is_pdf_document:
                selected_binary, page_info = apply_page_selection(
                    pdf_bytes=document_bytes,
                    page_mode=request.extraction.page_mode,
                    head_pages=request.extraction.head_pages,
                    tail_pages=request.extraction.tail_pages,
                )
                recorder.push("PAGE_SELECTION", "OK", "Seleccion de paginas aplicada.", page_info)
                update_job_progress(db, job_id, recorder, "RUNNING", "PAGE_SELECTION")

                probe = probe_pdf_extractability(selected_binary, request.extraction.probe_max_pages)
                recorder.push("PROBE", "OK", "Probe de extraibilidad completado.", probe)
                update_job_progress(db, job_id, recorder, "RUNNING", "PROBE")

                engine = resolve_extraction_engine(request.extraction, probe)
            else:
                page_info = {
                    "mode": "full",
                    "selected_pages": None,
                    "total_pages": None,
                    "reason": "non_pdf_docling_mode",
                }
                probe = {
                    "mime_type": mime_type,
                    "is_pdf": False,
                    "extractable_confidence": None,
                    "is_text_extractable": None,
                }
                recorder.push("PAGE_SELECTION", "SKIPPED", "Seleccion por paginas aplica solo a PDF.", page_info)
                recorder.push("PROBE", "SKIPPED", "Probe PyMuPDF aplica solo a PDF.", probe)
                update_job_progress(db, job_id, recorder, "RUNNING", "PROBE")
                requested_engine = safe_str(request.extraction.engine, "auto").strip().lower()
                if requested_engine == "pymupdf":
                    raise PipelineError(
                        "MIME_VALIDATION",
                        "INVALID_ENGINE_FOR_MIME",
                        "extraction.engine=pymupdf solo aplica para application/pdf.",
                        {"mime_type": mime_type, "engine": requested_engine},
                    )
                engine = "docling"

            extraction_meta: Dict[str, Any] = {"engine": engine, "mime_type": mime_type}
            if engine == "pymupdf":
                extracted_text, extracted_pages = extract_text_pymupdf(selected_binary)
                extraction_meta["ocr_confidence"] = {
                    "source": "pymupdf_probe",
                    "extractable_confidence": safe_float(probe.get("extractable_confidence"), None),
                    "is_text_extractable": safe_bool(probe.get("is_text_extractable"), False),
                    "pages_sampled": safe_int(probe.get("sample_pages"), None),
                    "text_page_ratio": safe_float(probe.get("text_page_ratio"), None),
                    "image_ratio": safe_float(probe.get("image_ratio"), None),
                }
            else:
                extracted_text, extracted_pages, docling_meta = extract_text_docling(
                    selected_binary,
                    request.extraction,
                    mime_type=mime_type,
                    file_name=file_name,
                )
                extraction_meta.update(docling_meta)
            if not extracted_text.strip():
                raise PipelineError(
                    "TEXT_EXTRACTION",
                    "EMPTY_EXTRACTED_TEXT",
                    "No se pudo extraer texto del archivo.",
                    {"engine": engine},
                )
            extraction_meta.update(
                {"chars": len(extracted_text), "pages_processed": extracted_pages, "engine_used": engine}
            )
            extraction_meta["ocr_quality"] = safe_float(
                extraction_meta.get("ocr_quality"),
                safe_float(probe.get("extractable_confidence"), None),
            )
            recorder.push("TEXT_EXTRACTION", "OK", "Extraccion completada.", extraction_meta)
            update_job_progress(db, job_id, recorder, "RUNNING", "TEXT_EXTRACTION")

            cleaned_text, cleaning_meta = clean_text(extracted_text, request.cleaning)
            text_for_chunking = cleaned_text if request.cleaning.enabled else extracted_text
            if not text_for_chunking.strip():
                raise PipelineError("TEXT_CLEANING", "EMPTY_AFTER_CLEANING", "Texto vacio despues de limpieza.")
            recorder.push("TEXT_CLEANING", "OK", "Limpieza aplicada.", cleaning_meta)
            update_job_progress(db, job_id, recorder, "RUNNING", "TEXT_CLEANING")

            # Persistencia temprana del OCR (texto extraido) en GestorDocumental.Documentos.
            ocr_words = count_words(text_for_chunking)
            ocr_text_hash = hashlib.sha256(text_for_chunking.encode("utf-8", errors="replace")).hexdigest()
            ocr_quality = safe_float(
                extraction_meta.get("ocr_quality"),
                safe_float(probe.get("extractable_confidence"), None),
            )
            ocr_pages = safe_int(extraction_meta.get("pages_processed"), None)
            ocr_updated_by = (
                request.created_by
                if request.created_by is not None
                else request.embedding.created_by_default
            )
            ocr_update_info: Optional[Dict[str, Any]] = None
            document_finalize_info: Optional[Dict[str, Any]] = None
            if documento_id is not None:
                ocr_update_info = db.update_documento_ocr_text(
                    documento_id=documento_id,
                    contenido_texto=text_for_chunking,
                    contenido_hash=binary_sha256,
                    calidad_ocr=ocr_quality,
                    paginas=ocr_pages,
                    palabras=ocr_words,
                    updated_by=ocr_updated_by,
                    estado="EN_PROCESAMIENTO",
                )
                if ocr_update_info is None:
                    raise PipelineError(
                        "OCR_PERSIST",
                        "DOCUMENT_NOT_UPDATED",
                        "No fue posible actualizar contenidoTexto para el documento resuelto.",
                        {"documento_id": documento_id, "service_endpoint": service_endpoint},
                    )
                recorder.push(
                    "OCR_PERSIST",
                    "OK",
                    "Texto OCR actualizado en GestorDocumental.Documentos.contenidoTexto.",
                    {
                        "documento_id": documento_id,
                        "chars": len(text_for_chunking),
                        "palabras": ocr_words,
                        "estado_documento": safe_str(ocr_update_info.get("estado_documento"), ""),
                        "service_endpoint": service_endpoint,
                    },
                )
            else:
                recorder.push(
                    "OCR_PERSIST",
                    "WARN",
                    "No se actualiza contenidoTexto: documento_id_resuelto es null.",
                    {"oid": int(request.oid), "service_endpoint": service_endpoint},
                )
            update_job_progress(
                db,
                job_id,
                recorder,
                "RUNNING",
                "OCR_PERSIST",
                {
                    "service_endpoint": service_endpoint,
                    "documento_id_resuelto": documento_id,
                    "ocr_chars": len(text_for_chunking),
                    "ocr_words": ocr_words,
                    "binary_sha256": binary_sha256,
                    "ocr_text_hash": ocr_text_hash,
                },
            )

            chunks: List[str] = []
            bounds: List[Tuple[int, int]] = []
            chunking_method = "none"
            chunk_summary: Optional[Dict[str, Any]] = None
            if run_chunking_stage:
                chunk_strategy = safe_str(request.chunking.strategy, "semantic").strip().lower()
                if chunk_strategy not in {"semantic", "simple"}:
                    raise PipelineError(
                        "CHUNKING",
                        "INVALID_CHUNKING_STRATEGY",
                        "chunking.strategy debe ser semantic o simple.",
                        {"strategy": request.chunking.strategy},
                )
                chunking_method = chunk_strategy
                if chunk_strategy == "semantic":
                    semantic_max_tokens = max(256, derive_embedding_max_length(request.chunking))
                    tokenizer_for_chunk: Optional[Any] = None
                    try:
                        tokenizer_for_chunk = load_tokenizer(request.embedding.model_name)
                    except Exception:
                        tokenizer_for_chunk = None
                    try:
                        chunks = semantic_chunk_text(
                            text_for_chunking,
                            tokenizer_for_chunk,
                            request.embedding.model_name,
                            semantic_max_tokens,
                        )
                    except Exception as semantic_exc:
                        semantic_error = f"{type(semantic_exc).__name__}: {str(semantic_exc)}"
                        force_simple = (
                            "max_tokens could not be determined automatically" in semantic_error.lower()
                            or "sentence_bert_config.json" in semantic_error.lower()
                        )
                        if force_simple or request.chunking.enable_simple_fallback:
                            chunks = simple_chunk_text(
                                text_for_chunking,
                                request.chunking.simple_chunk_size,
                                request.chunking.simple_chunk_overlap,
                            )
                            chunking_method = (
                                "simple_forced_semantic_runtime"
                                if force_simple
                                else "simple_fallback"
                            )
                            recorder.push(
                                "CHUNKING_WARNING",
                                "WARN",
                                "Chunking semantico fallo; se aplico chunking simple.",
                                {
                                    "reason": semantic_error,
                                    "force_simple": force_simple,
                                    "max_tokens": semantic_max_tokens,
                                },
                            )
                        else:
                            raise PipelineError(
                                "CHUNKING",
                                "SEMANTIC_CHUNKING_FAILED",
                                "Fallo chunking semantico.",
                                {"error": semantic_error, "max_tokens": semantic_max_tokens},
                            ) from semantic_exc
                else:
                    chunks = simple_chunk_text(
                        text_for_chunking,
                        request.chunking.simple_chunk_size,
                        request.chunking.simple_chunk_overlap,
                    )

                if request.chunking.max_chunks > 0:
                    chunks = chunks[: int(request.chunking.max_chunks)]
                chunks = [chunk for chunk in chunks if chunk.strip()]
                if not chunks:
                    raise PipelineError("CHUNKING", "EMPTY_CHUNKS", "No se generaron chunks.")
                if len(text_for_chunking) < int(request.chunking.min_text_chars):
                    recorder.push(
                        "CHUNKING_WARNING",
                        "WARN",
                        "Texto menor al umbral min_text_chars; se continua con los chunks disponibles.",
                        {
                            "min_text_chars": int(request.chunking.min_text_chars),
                            "text_chars": len(text_for_chunking),
                        },
                    )
                bounds = estimate_bounds(text_for_chunking, chunks)
                chunk_lengths = [len(c) for c in chunks]
                short_threshold = 50
                empty_chunks = sum(1 for c in chunks if not c.strip())
                short_chunks = sum(1 for ln in chunk_lengths if 0 < ln < short_threshold)
                chunk_summary = {
                    "strategy": chunking_method,
                    "chunks_count": len(chunks),
                    "chars_total": sum(chunk_lengths),
                    "chars_min": min(chunk_lengths) if chunk_lengths else 0,
                    "chars_max": max(chunk_lengths) if chunk_lengths else 0,
                    "chars_avg": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0,
                    "empty_chunks": empty_chunks,
                    "short_chunks": short_chunks,
                    "short_threshold": short_threshold,
                }
                if empty_chunks > 0 or short_chunks > 0:
                    recorder.push(
                        "CHUNKING_QUALITY",
                        "WARN",
                        f"Detectados {empty_chunks} chunks vacíos y {short_chunks} chunks cortos (<{short_threshold} chars).",
                        chunk_summary,
                    )
                recorder.push(
                    "CHUNKING",
                    "OK",
                    "Chunking completado.",
                    chunk_summary,
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "CHUNKING")
            else:
                recorder.push("CHUNKING", "SKIPPED", "Etapa no solicitada para este endpoint.")

            vectors: List[List[float]] = []
            tokens_per_chunk: List[int] = [0 for _ in chunks]
            embedding_device = "none"
            if run_embedding_stage:
                tokenizer, model, device = load_embedding_model(request.embedding.model_name)
                derived_max_length = derive_embedding_max_length(request.chunking)
                vectors, tokens_per_chunk = embed_chunks(
                    chunks=chunks,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=derived_max_length,
                    batch_size=request.embedding.batch_size,
                )
                embedding_device = device
                if len(vectors) != len(chunks):
                    raise PipelineError(
                        "EMBEDDINGS",
                        "EMBEDDING_COUNT_MISMATCH",
                        "La cantidad de embeddings no coincide con chunks.",
                        {"chunks": len(chunks), "vectors": len(vectors)},
                    )
                recorder.push(
                    "EMBEDDINGS",
                    "OK",
                    "Embeddings generados.",
                    {"enabled": True, "device": embedding_device, "vectors": len(vectors)},
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "EMBEDDINGS")
            else:
                recorder.push("EMBEDDINGS", "SKIPPED", "Etapa no solicitada para este endpoint.")

            inserted_rows = 0
            if run_persist_stage:
                existing_count = 0
                deleted_count = 0
                if documento_id is not None:
                    existing_count = db.count_existing_embeddings(documento_id=documento_id)
                    if existing_count > 0 and not request.overwrite.enabled:
                        raise PipelineError(
                            "OVERWRITE_CHECK",
                            "DUPLICATE_EMBEDDINGS",
                            "Ya existen embeddings para el documento y overwrite.enabled=false.",
                            {"documento_id": documento_id, "existing_count": existing_count},
                        )
                    if existing_count > 0 and request.overwrite.enabled:
                        deleted_count = db.delete_existing_embeddings(documento_id=documento_id)
                else:
                    recorder.push(
                        "OVERWRITE_CHECK",
                        "WARN",
                        "documentoId no resuelto; no se valida/borra duplicados por documento.",
                        {"oid": int(request.oid), "overwrite_enabled": bool(request.overwrite.enabled)},
                    )

                recorder.push(
                    "OVERWRITE_CHECK",
                    "OK",
                    "Validacion de duplicados completada.",
                    {
                        "existing_count": existing_count,
                        "overwrite_enabled": bool(request.overwrite.enabled),
                        "deleted_count": deleted_count,
                    },
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "OVERWRITE_CHECK")

                created_by = request.created_by if request.created_by is not None else request.embedding.created_by_default
                row_values: List[Tuple[Any, ...]] = []
                for idx, chunk in enumerate(chunks):
                    chunk_inicio, chunk_fin = bounds[idx]
                    vector_literal = vector_to_pg_literal(vectors[idx])
                    metadata_row = {
                        "oid": int(request.oid),
                        "file_name": file_name,
                        "usuario_proceso": request.usuario_proceso,
                        "job_proceso": request.job_proceso,
                        "phase": "PERSIST",
                        "item_id": safe_int(item_info.get("item_id"), None) if item_info else None,
                        "probe": probe,
                        "engine_used": engine,
                        "ocr_quality": safe_float(extraction_meta.get("ocr_quality"), None),
                        "ocr_confidence": to_json_safe(
                            extraction_meta.get("docling_confidence")
                            or extraction_meta.get("ocr_confidence")
                            or {}
                        ),
                        "documento_id_resuelto": documento_id,
                        "documento_info": to_json_safe(documento_info or {}),
                        "oid_stats": to_json_safe(oid_stats or {}),
                        "page_selection": page_info,
                        "chunk_strategy": chunking_method,
                        "chunk_chars": len(chunk),
                        "request_metadata_input": request.metadata,
                    }
                    row_values.append(
                        (
                            int(chunk_fin),
                            int(idx + 1),
                            int(chunk_inicio),
                            chunk,
                            int(created_by),
                            documento_id,
                            request.job_filde_id,
                            json_dumps_safe(metadata_row),
                            request.embedding.model_name,
                            int(tokens_per_chunk[idx]) if idx < len(tokens_per_chunk) else 0,
                            vector_literal,
                        )
                    )

                try:
                    inserted_rows = db.insert_embeddings(row_values)
                except psycopg2.errors.ForeignKeyViolation as exc:
                    raise PipelineError(
                        "PERSIST",
                        "INVALID_DOCUMENTO_REFERENCE",
                        "No fue posible insertar embeddings por referencia documentoId invalida.",
                        {
                            "documento_id_resuelto": documento_id,
                            "oid": int(request.oid),
                            "hint": "Valide mapeo entre OID y GestorDocumental.Documentos.id.",
                            "db_error": str(exc),
                        },
                    ) from exc
                except psycopg2.errors.NotNullViolation as exc:
                    raise PipelineError(
                        "PERSIST",
                        "MISSING_REQUIRED_DOCUMENTO_ID",
                        "La tabla de embeddings exige documentoId y no fue posible resolverlo.",
                        {
                            "documento_id_resuelto": documento_id,
                            "oid": int(request.oid),
                            "hint": (
                                "Registre/actualice el documento en GestorDocumental.Documentos con "
                                "metadatosExtra.ocr.metadata.oid o archivoNombre trazable."
                            ),
                            "db_error": str(exc),
                        },
                    ) from exc
                if request.embedding.require_inserted_rows and inserted_rows <= 0:
                    raise PipelineError(
                        "PERSIST",
                        "NO_ROWS_INSERTED",
                        "No se insertaron embeddings y require_inserted_rows=true.",
                    )
                recorder.push(
                    "PERSIST",
                    "OK",
                    "Persistencia completada.",
                    {"inserted_rows": inserted_rows, "save_to_db": True},
                )
                update_job_progress(db, job_id, recorder, "RUNNING", "PERSIST")
            else:
                recorder.push("PERSIST", "SKIPPED", "Persistencia no solicitada para este endpoint.")

            # Finaliza estado documental al completar chunking + embeddings.
            if run_embedding_stage:
                if documento_id is not None:
                    metadata_patch = {
                        "ocr_embedding_pipeline": {
                            "service_endpoint": service_endpoint,
                            "stage": stage_key,
                            "job_id": job_id,
                            "job_filde_id": request.job_filde_id,
                            "usuario_proceso": request.usuario_proceso,
                            "job_proceso": request.job_proceso,
                            "updated_at_utc": utc_now_iso(),
                            "ocr": {
                                "engine": engine,
                                "chars": len(text_for_chunking),
                                "palabras": ocr_words,
                                "paginas": ocr_pages,
                                "calidad_ocr": ocr_quality,
                                "hash_binario": binary_sha256,
                                "hash_texto_limpio": ocr_text_hash,
                                "probe": to_json_safe(probe),
                                "confianza": to_json_safe(
                                    extraction_meta.get("docling_confidence")
                                    or extraction_meta.get("ocr_confidence")
                                    or {}
                                ),
                            },
                            "chunking": {
                                "strategy": chunking_method,
                                "chunks_count": len(chunks),
                            },
                            "embedding": {
                                "enabled": bool(run_embedding_stage),
                                "model_name": request.embedding.model_name,
                                "vectors_count": len(vectors),
                                "save_to_db": bool(run_persist_stage),
                                "inserted_rows": int(inserted_rows),
                            },
                        }
                    }
                    document_finalize_info = db.update_documento_embedding_completion(
                        documento_id=documento_id,
                        metadata_patch=metadata_patch,
                        updated_by=ocr_updated_by,
                    )
                    if document_finalize_info is None:
                        raise PipelineError(
                            "DOCUMENT_FINALIZE",
                            "DOCUMENT_NOT_UPDATED",
                            "No fue posible actualizar metadatosExtra al finalizar embeddings/chunking.",
                            {"documento_id": documento_id, "service_endpoint": service_endpoint},
                        )
                    recorder.push(
                        "DOCUMENT_FINALIZE",
                        "OK",
                        "metadatosExtra actualizado. Flags (embeddingGenerado, estado) los gestiona el consumer.",
                        {
                            "documento_id": documento_id,
                            "estado_documento": safe_str(document_finalize_info.get("estado_documento"), ""),
                            "embedding_generado": safe_bool(document_finalize_info.get("embedding_generado"), False),
                            "service_endpoint": service_endpoint,
                        },
                    )
                else:
                    recorder.push(
                        "DOCUMENT_FINALIZE",
                        "WARN",
                        "No se marca documento como PROCESADO: documento_id_resuelto es null.",
                        {"oid": int(request.oid), "service_endpoint": service_endpoint},
                    )
                update_job_progress(db, job_id, recorder, "RUNNING", "DOCUMENT_FINALIZE")
            else:
                recorder.push(
                    "DOCUMENT_FINALIZE",
                    "SKIPPED",
                    "Actualizacion final de documento aplica solo en etapas con embeddings.",
                )

            elapsed_ms = int((time.monotonic() - started) * 1000)
            result_data = {
                "oid": int(request.oid),
                "stage": stage_key,
                "job_id": job_id,
                "file_name": file_name,
                "item_id": safe_int(item_info.get("item_id"), None) if item_info else None,
                "oid_documento": int(request.oid),
                "documento_id_resuelto": documento_id,
                "documento_info": to_json_safe(documento_info or {}),
                "oid_stats": to_json_safe(oid_stats or {}),
                "job_filde_id": request.job_filde_id,
                "usuario_proceso": request.usuario_proceso,
                "job_proceso": request.job_proceso,
                "service_endpoint": service_endpoint,
                "mime_type": mime_type,
                "mime_validation": to_json_safe(mime_support_info),
                "queue_name": queue_name if request.queue.enabled else None,
                "binary_sha256": binary_sha256,
                "engine_used": engine,
                "ocr_quality": safe_float(extraction_meta.get("ocr_quality"), None),
                "ocr_confidence": to_json_safe(
                    extraction_meta.get("docling_confidence")
                    or extraction_meta.get("ocr_confidence")
                    or {}
                ),
                "probe": probe,
                "page_selection": page_info,
                "cleaning": cleaning_meta,
                "ocr_document_update": to_json_safe(ocr_update_info or {}),
                "document_finalize_update": to_json_safe(document_finalize_info or {}),
                "chunks_count": len(chunks),
                "chunks_summary": to_json_safe(chunk_summary) if chunk_summary else None,
                "inserted_rows": inserted_rows,
                "gpu": gpu_metrics(),
                "elapsed_ms": elapsed_ms,
            }
            if stage_key == "ocr":
                result_data["ocr_text_chars"] = len(text_for_chunking)
                result_data["ocr_preview"] = text_for_chunking[:500]
            if stage_key == "chunking":
                result_data["chunks_preview"] = chunks[:5]
            if request.embedding.return_vectors and vectors:
                result_data["vectors"] = vectors

            final_payload = {
                "pipeline_status": "COMPLETED",
                "current_phase": "DONE",
                "phases": recorder.as_list(),
                "result": result_data,
                "updated_at_utc": utc_now_iso(),
            }
            db.update_job_state(
                job_id=job_id,
                estado="COMPLETADO",
                resultado=final_payload,
                error_message=None,
                clear_error=True,
                set_fin=True,
            )

            try:
                with PostgresClient(PostgresSettings.from_env()) as db_log:
                    db_log.log_to_operaciones(
                        nivel="INFO", modulo="ocr_chunking",
                        mensaje=f"Pipeline completado: oid={int(request.oid)} doc={file_name}"[:2000],
                        contexto=json.dumps({"oid": int(request.oid), "job_id": job_id, "file_name": file_name, "stage": stage_key}, ensure_ascii=False),
                    )
            except Exception:
                LOGGER.debug("No fue posible registrar exito en Operaciones.", exc_info=True)

            return OCRChunkingResponse(
                status="COMPLETED",
                exitoso=True,
                message=f"Proceso '{stage_key}' completado exitosamente.",
                phases=recorder.as_list(),
                data=result_data,
            )

    except PipelineError as exc:
        LOGGER.error("PipelineError phase=%s code=%s message=%s", exc.phase, exc.code, exc.message)
        trace_text = traceback.format_exc()
        try:
            with PostgresClient(PostgresSettings.from_env()) as db_log:
                db_log.log_to_operaciones(
                    nivel="ERROR", modulo="ocr_chunking",
                    mensaje=f"[{exc.phase}] {exc.code}: {exc.message}"[:2000],
                    contexto=json.dumps({"phase": exc.phase, "code": exc.code, "oid": int(request.oid), "job_id": job_id, "file_name": file_name, "retryable": exc.retryable}, ensure_ascii=False),
                    stack_trace=trace_text[:4000],
                    also_error=True, error_tipo=exc.code,
                )
        except Exception:
            LOGGER.debug("No fue posible registrar PipelineError en Operaciones.", exc_info=True)
        error_details = dict(exc.details or {})
        if not error_details.get("traceback"):
            error_details["traceback"] = trace_text
        recorder.push(exc.phase, "ERROR", exc.message, error_details)
        error_payload = {
            "phase": exc.phase,
            "code": exc.code,
            "message": exc.message,
            "details": error_details,
            "retryable": exc.retryable,
        }
        if job_id is not None:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_error:
                    db_error.update_job_state(
                        job_id=job_id,
                        estado="ERROR",
                        resultado={
                            "pipeline_status": "FAILED",
                            "current_phase": exc.phase,
                            "phases": recorder.as_list(),
                            "error": error_payload,
                            "updated_at_utc": utc_now_iso(),
                        },
                        error_message=exc.message,
                        set_fin=True,
                    )
            except Exception:
                LOGGER.exception("No fue posible actualizar job en estado ERROR.")

        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Proceso fallido.",
            error=error_payload,
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key, "job_id": job_id, "file_name": file_name},
        )

    except Exception as exc:
        LOGGER.exception("Error inesperado en pipeline.")
        try:
            with PostgresClient(PostgresSettings.from_env()) as db_log:
                db_log.log_to_operaciones(
                    nivel="CRITICAL", modulo="ocr_chunking",
                    mensaje=f"UNHANDLED: {type(exc).__name__}: {str(exc)}"[:2000],
                    contexto=json.dumps({"oid": int(request.oid), "job_id": job_id, "file_name": file_name}, ensure_ascii=False),
                    stack_trace=traceback.format_exc()[:4000],
                    also_error=True, error_tipo="UNEXPECTED_ERROR",
                )
        except Exception:
            LOGGER.debug("No fue posible registrar error inesperado en Operaciones.", exc_info=True)
        unknown = PipelineError(
            "UNHANDLED_EXCEPTION",
            "UNEXPECTED_ERROR",
            f"{type(exc).__name__}: {str(exc)}",
            {"traceback": traceback.format_exc()},
        )
        recorder.push(unknown.phase, "ERROR", unknown.message, unknown.details)
        if job_id is not None:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_error:
                    db_error.update_job_state(
                        job_id=job_id,
                        estado="ERROR",
                        resultado={
                            "pipeline_status": "FAILED",
                            "current_phase": unknown.phase,
                            "phases": recorder.as_list(),
                            "error": unknown.to_dict(),
                            "updated_at_utc": utc_now_iso(),
                        },
                        error_message=unknown.message,
                        set_fin=True,
                    )
            except Exception:
                LOGGER.exception("No fue posible actualizar job en estado ERROR.")

        return OCRChunkingResponse(
            status="FAILED",
            exitoso=False,
            message="Proceso fallido por error inesperado.",
            error=unknown.to_dict(),
            phases=recorder.as_list(),
            data={"oid": int(request.oid), "stage": stage_key, "job_id": job_id, "file_name": file_name},
        )

    finally:
        # Liberar memoria GPU/CPU despues de cada ejecucion del pipeline
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if request.queue.enabled and queue_slot_acquired:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_final:
                    db_final.release_queue_slot(queue_name)
                    db_final.refresh_queue_stats(queue_name)
            except Exception:
                LOGGER.exception("No fue posible liberar/actualizar queue.")
        elif request.queue.enabled:
            try:
                with PostgresClient(PostgresSettings.from_env()) as db_final:
                    db_final.refresh_queue_stats(queue_name)
            except Exception:
                LOGGER.exception("No fue posible refrescar queue stats.")


def process_request(request: OCRChunkingRequest, stage: str = "pipeline") -> OCRChunkingResponse:
    """Selects mock or real pipeline execution by stage. Writes structured log."""
    if request.mock.enabled:
        return run_mock_pipeline(request)
    response = run_real_pipeline(request, stage=stage)
    try:
        write_pipeline_log(
            oid=int(request.oid),
            stage=stage,
            status=response.status,
            phases=response.phases,
            data=response.data,
            error=response.error,
            source="pipeline",
        )
    except Exception:
        LOGGER.debug("No se pudo escribir log de pipeline para oid=%s", request.oid, exc_info=True)
    return response

def process_batch(request: OCRChunkingBatchRequest, stage: str = "pipeline") -> OCRChunkingBatchResponse:
    """Processes a batch of requests sequentially or in parallel."""
    total = len(request.requests)
    results: List[Optional[OCRChunkingResponse]] = [None] * total

    if request.parallel_workers <= 1:
        for idx, item in enumerate(request.requests):
            results[idx] = process_request(item, stage=stage)
    else:
        with ThreadPoolExecutor(max_workers=request.parallel_workers) as pool:
            future_map = {pool.submit(process_request, item, stage): idx for idx, item in enumerate(request.requests)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = OCRChunkingResponse(
                        status="FAILED",
                        exitoso=False,
                        message="Error inesperado en ejecucion batch.",
                        error={
                            "phase": "BATCH",
                            "code": "BATCH_UNHANDLED_EXCEPTION",
                            "message": f"{type(exc).__name__}: {str(exc)}",
                        },
                        phases=[],
                        data={"oid": int(request.requests[idx].oid)},
                    )

    final_results = [r for r in results if r is not None]
    completed = sum(1 for r in final_results if r.exitoso)
    failed = total - completed
    overall_ok = failed == 0
    return OCRChunkingBatchResponse(
        status="COMPLETED" if overall_ok else "PARTIAL",
        exitoso=overall_ok,
        message="Batch procesado." if overall_ok else "Batch procesado con errores.",
        total=total,
        completados=completed,
        fallidos=failed,
        resultados=final_results,  # type: ignore[arg-type]
    )


def sample_request() -> Dict[str, Any]:
    """Returns canonical sample request."""
    model = OCRChunkingRequest(
        oid=2299268,
        nombre_documento="CTO_EyP_LLA_50_2013.pdf",
        file_name=None,
        mime_type="application/pdf",
        job_filde_id=4567,
        usuario_proceso="analista_anh",
        job_proceso="JOB_OCR_20260309_001",
        created_by=1101,
        metadata={
            "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
            "metadata_documento": {"paginas": 133, "idioma": "es"},
            "ruta_pdf": "\\\\servidor\\share\\CTO_EyP_LLA_50_2013.pdf",
        },
        queue=QueueOptions(enabled=True, max_concurrency=2, queue_when_busy=True),
        overwrite=OverwriteOptions(enabled=False, allow_duplicate_hash=False, allow_reprocess_processed=False),
        extraction=ExtractionOptions(
            engine="auto",
            enable_pymupdf_fast_path=True,
            fast_path_confidence_threshold=0.85,
            page_mode="head_tail",
            head_pages=8,
            tail_pages=8,
        ),
        cleaning=CleaningOptions(enabled=True, remove_headers=True, remove_isolated_numbers=True),
        chunking=ChunkingOptions(strategy="semantic", max_chunks=0, enable_simple_fallback=False),
        embedding=EmbeddingOptions(enabled=True, model_name=DEFAULT_EMBEDDING_MODEL, save_to_db=True, return_vectors=False),
        mock=MockOptions(enabled=False),
    )
    return pydantic_model_dump(model)


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    openapi_tags=OPENAPI_TAGS,
    description=(
        "Servicio OpenAPI para OCR, chunking y embeddings.\n"
        "Rutas funcionales: /ocr-docling, /chunking-docling, /embedding-generation, /PipelineOCR.\n"
        "Entrada obligatoria: oid.\n"
        "Acceso protegido por JWT Bearer validado contra Keycloak JWKS."
    ),
)


@app.get("/health", tags=[TAG_HELPERS])
def health(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """Health endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp_utc": utc_now_iso(),
        "gpu": gpu_metrics(),
    }


@app.get("/health/dependencies", tags=[TAG_HELPERS])
def health_dependencies(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """Verifica dependencias externas del servicio."""
    import shutil

    # --- LibreOffice ---
    lo_cmd = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo_cmd:
        for candidate in [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/usr/local/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/snap/bin/libreoffice",
        ]:
            if os.path.isfile(candidate):
                lo_cmd = candidate
                break

    lo_version = None
    if lo_cmd:
        try:
            import subprocess
            proc = subprocess.run(
                [lo_cmd, "--version"], capture_output=True, text=True, timeout=10,
            )
            lo_version = proc.stdout.strip() or proc.stderr.strip() or "desconocida"
        except Exception as exc:
            LOGGER.error("LibreOffice version check failed: %s", exc)
            lo_version = "error: no se pudo obtener la version"

    # --- PostgreSQL ---
    pg_ok = False
    pg_error = None
    try:
        with PostgresClient(PostgresSettings.from_env()) as db:
            db.execute_returning_one("SELECT 1", ())
            pg_ok = True
    except Exception as exc:
        LOGGER.error("PostgreSQL health check failed: %s", exc)
        pg_error = "Database connection failed"

    # --- GPU / CUDA ---
    gpu_info = gpu_metrics()

    # --- Docling ---
    docling_version = None
    try:
        import docling
        docling_version = getattr(docling, "__version__", "instalado")
    except ImportError:
        docling_version = "NO INSTALADO"

    # --- Transformers (embeddings) ---
    transformers_version = None
    try:
        import transformers
        transformers_version = getattr(transformers, "__version__", "instalado")
    except ImportError:
        transformers_version = "NO INSTALADO"

    # --- PyTorch ---
    torch_info = {
        "instalado": torch.cuda.is_available() or True,
        "version": getattr(torch, "__version__", "desconocida"),
        "cuda": torch.cuda.is_available(),
    }

    deps = {
        "libreoffice": {
            "instalado": lo_cmd is not None,
            "version": lo_version,
            "formatos_legacy_soportados": [".doc", ".xls", ".ppt"] if lo_cmd else [],
            "nota": None if lo_cmd else "Archivos .doc/.xls/.ppt seran etiquetados como 'Formato no soportado'",
        },
        "postgresql": {
            "conectado": pg_ok,
            "error": pg_error,
        },
        "gpu": gpu_info,
        "docling": {
            "instalado": docling_version != "NO INSTALADO",
            "version": docling_version,
        },
        "transformers": {
            "instalado": transformers_version != "NO INSTALADO",
            "version": transformers_version,
        },
        "torch": torch_info,
    }

    all_ok = all([
        lo_cmd is not None,
        pg_ok,
        gpu_info.get("cuda_available", False),
        docling_version != "NO INSTALADO",
        transformers_version != "NO INSTALADO",
    ])

    return {
        "status": "ok" if all_ok else "degraded",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp_utc": utc_now_iso(),
        "dependencies": deps,
    }


@app.get("/example-request", tags=[TAG_HELPERS])
def example_request(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """Returns request payload example."""
    return {"input": sample_request()}


@app.get("/validate-db", tags=[TAG_HELPERS])
def validate_db(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Valida conexión a Postgres a nivel API.
    Retorna versión, metadatos y una consulta de prueba.
    """
    result = _validate_db_connection()
    result["timestamp"] = bogota_now_iso()
    result["service"] = SERVICE_NAME
    if safe_str(result.get("status"), "").lower() != "ok":
        raise HTTPException(status_code=403, detail=to_json_safe(result))
    return to_json_safe(result)


KNOWN_GPU_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "sm_50": {"family": "Maxwell",  "example": "GTX 950/960"},
    "sm_60": {"family": "Pascal",   "example": "GP100, Tesla P100"},
    "sm_61": {"family": "Pascal",   "example": "GTX 1080, Tesla P40"},
    "sm_70": {"family": "Volta",    "example": "Tesla V100"},
    "sm_75": {"family": "Turing",   "example": "RTX 2080, T4"},
    "sm_80": {"family": "Ampere",   "example": "A100"},
    "sm_86": {"family": "Ampere",   "example": "RTX 3090, RTX A6000"},
    "sm_89": {"family": "Ada Lovelace", "example": "RTX 4090, L40"},
    "sm_90": {"family": "Hopper",   "example": "H100, H200"},
    "sm_100": {"family": "Blackwell", "example": "B100, B200"},
    "sm_120": {"family": "Blackwell", "example": "RTX 5090, RTX 5080"},
}

REQUIRED_LIBRARIES: Dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "docling": "docling",
    "docling-core": "docling_core",
    "docling-ibm-models": "docling_ibm_models",
    "docling-parse": "docling_parse",
    "rapidocr": "rapidocr",
    "onnxruntime": "onnxruntime",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pymupdf": "fitz",
    "numpy": "numpy",
    "Pillow": "PIL",
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
    "psycopg2-binary": "psycopg2",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
}


def _get_lib_version(import_name: str) -> Optional[str]:
    """Returns installed version of a library or None."""
    try:
        mod = sys.modules.get(import_name) or __import__(import_name)
        ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
        if ver:
            return str(ver)
        from importlib.metadata import version as meta_version
        return meta_version(import_name)
    except Exception:
        try:
            from importlib.metadata import version as meta_version
            return meta_version(import_name)
        except Exception:
            return None


def _gpu_arch_info(compute_major: int, compute_minor: int) -> Dict[str, Any]:
    """Returns architecture details for a given compute capability."""
    sm = f"sm_{compute_major}{compute_minor}"
    info = KNOWN_GPU_ARCHITECTURES.get(sm, {})
    return {
        "compute_capability": f"{compute_major}.{compute_minor}",
        "sm_code": sm,
        "family": info.get("family", "unknown"),
        "example_gpus": info.get("example", "unknown"),
    }


@app.get("/validate-gpu", tags=[TAG_INFRA])
def validate_gpu(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Diagnostico completo de GPU: arquitectura, compute capability,
    compatibilidad con PyTorch y ONNX Runtime, memoria por dispositivo.
    Pensado para validar RTX A6000 (Ampere, sm_86) en produccion
    y RTX 5090 (Blackwell, sm_120) en desarrollo.
    """
    result: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "service": SERVICE_NAME,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        result["status"] = "error"
        result["message"] = "CUDA no disponible. Verificar driver NVIDIA y version de torch."
        return result

    torch_cuda_version = getattr(torch.version, "cuda", None)
    torch_arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    device_count = torch.cuda.device_count()

    devices = []
    all_compatible = True
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        cc_major, cc_minor = props.major, props.minor
        arch = _gpu_arch_info(cc_major, cc_minor)
        sm_code = arch["sm_code"]
        compatible = sm_code in torch_arch_list
        if not compatible:
            all_compatible = False

        total_mem = int(props.total_memory)
        allocated = int(torch.cuda.memory_allocated(i))
        reserved = int(torch.cuda.memory_reserved(i))

        devices.append({
            "index": i,
            "name": props.name,
            "architecture": arch,
            "compatible_with_torch": compatible,
            "memory": {
                "total_bytes": total_mem,
                "total_gb": round(total_mem / (1024 ** 3), 2),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "free_estimate_bytes": max(total_mem - reserved, 0),
            },
            "multi_processor_count": props.multi_processor_count,
        })

    ort_providers: List[str] = []
    ort_version: Optional[str] = None
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        ort_providers = ort.get_available_providers()
    except ImportError:
        pass

    result.update({
        "status": "ok" if all_compatible else "warning",
        "message": (
            "Todas las GPU son compatibles con PyTorch."
            if all_compatible
            else "Una o mas GPU no tienen kernels en esta build de PyTorch. "
                 "Verificar version CUDA de torch (cu126/cu128/cu130)."
        ),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "supported_architectures": torch_arch_list,
        },
        "onnxruntime": {
            "version": ort_version,
            "available_providers": ort_providers,
            "gpu_enabled": "CUDAExecutionProvider" in ort_providers,
        },
        "device_count": device_count,
        "devices": devices,
    })
    return result


@app.get("/validate-libraries", tags=[TAG_INFRA])
def validate_libraries(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Reporta versiones instaladas de todas las librerias criticas del servicio.
    Util para comparar entornos (dev vs produccion) y detectar discrepancias.
    """
    libraries: List[Dict[str, Any]] = []
    missing: List[str] = []

    for pkg_name, import_name in REQUIRED_LIBRARIES.items():
        version = _get_lib_version(import_name)
        if version:
            libraries.append({"package": pkg_name, "import": import_name, "version": version})
        else:
            missing.append(pkg_name)
            libraries.append({"package": pkg_name, "import": import_name, "version": None})

    torch_version = torch.__version__ if hasattr(torch, "__version__") else None

    return {
        "timestamp_utc": utc_now_iso(),
        "service": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "status": "ok" if not missing else "warning",
        "message": (
            "Todas las librerias estan instaladas."
            if not missing
            else f"Librerias no encontradas: {', '.join(missing)}"
        ),
        "torch_build": torch_version,
        "libraries": libraries,
        "missing": missing,
    }


@app.get("/validate-cuda-stress", tags=[TAG_INFRA])
def validate_cuda_stress(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Ejecuta una prueba rapida de tensor en cada GPU para verificar
    que los kernels CUDA funcionan correctamente (no solo que se detectan).
    Detecta el error 'no kernel image is available' antes de procesar documentos.
    """
    if not torch.cuda.is_available():
        return {
            "timestamp_utc": utc_now_iso(),
            "status": "error",
            "message": "CUDA no disponible.",
            "devices": [],
        }

    device_count = torch.cuda.device_count()
    results = []
    all_ok = True

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        test_result: Dict[str, Any] = {"index": i, "name": device_name}
        try:
            t = torch.randn(256, 256, device=f"cuda:{i}")
            out = torch.mm(t, t)
            _ = out.sum().item()
            del t, out
            torch.cuda.empty_cache()
            test_result["status"] = "ok"
            test_result["message"] = "Operacion matmul 256x256 exitosa."
        except Exception as exc:
            all_ok = False
            test_result["status"] = "error"
            test_result["message"] = str(exc)
            test_result["error_type"] = type(exc).__name__
        results.append(test_result)

    return {
        "timestamp_utc": utc_now_iso(),
        "status": "ok" if all_ok else "error",
        "message": (
            "Todas las GPU pasaron la prueba de stress CUDA."
            if all_ok
            else "Una o mas GPU fallaron. Revisar compatibilidad torch/CUDA/driver."
        ),
        "device_count": device_count,
        "devices": results,
    }


@app.get("/validate-environment", tags=[TAG_INFRA])
def validate_environment(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Resumen ejecutivo del entorno: sistema operativo, Python, GPU,
    estado general de librerias y CUDA. Un solo endpoint para
    validar rapidamente si un nodo esta listo para operar.
    """
    gpu_ok = False
    gpu_summary: List[Dict[str, str]] = []
    cuda_functional = False

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            arch = _gpu_arch_info(props.major, props.minor)
            compatible = arch["sm_code"] in (torch.cuda.get_arch_list() or [])
            gpu_summary.append({
                "index": str(i),
                "name": props.name,
                "family": arch["family"],
                "sm_code": arch["sm_code"],
                "compatible": str(compatible),
                "vram_gb": str(round(props.total_memory / (1024 ** 3), 2)),
            })
            if compatible:
                gpu_ok = True

        try:
            t = torch.randn(16, 16, device="cuda:0")
            _ = (t @ t).sum().item()
            del t
            cuda_functional = True
        except Exception:
            cuda_functional = False

    ort_gpu = False
    try:
        import onnxruntime as ort
        ort_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass

    missing_libs = []
    for pkg_name, import_name in REQUIRED_LIBRARIES.items():
        if _get_lib_version(import_name) is None:
            missing_libs.append(pkg_name)

    all_ok = gpu_ok and cuda_functional and ort_gpu and len(missing_libs) == 0

    return {
        "timestamp_utc": utc_now_iso(),
        "service": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "status": "ok" if all_ok else "warning",
        "ready_for_production": all_ok,
        "system": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "hostname": platform.node(),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "functional": cuda_functional,
            "torch_build": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        },
        "onnxruntime_gpu": ort_gpu,
        "gpu_devices": gpu_summary,
        "missing_libraries": missing_libs,
        "checks": {
            "gpu_detected": torch.cuda.is_available(),
            "gpu_arch_compatible": gpu_ok,
            "cuda_kernels_functional": cuda_functional,
            "onnxruntime_gpu_available": ort_gpu,
            "all_libraries_installed": len(missing_libs) == 0,
        },
    }


# ---------------------------------------------------------------------------
# Structured file logging system (JSONL)
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = os.path.join(tempfile.gettempdir(), "ocr-chunking-logs")
LOG_DIR = os.getenv("OCR_LOG_DIR", DEFAULT_LOG_DIR)
LOG_RETENTION_DAYS = int(os.getenv("OCR_LOG_RETENTION_DAYS", "30"))
_LOG_WRITE_LOCK = Lock()


def _ensure_log_dir(subdir: str = "") -> str:
    """Creates log directory (cross-platform). Returns the path."""
    target = os.path.join(LOG_DIR, "ocr-chunking", subdir) if subdir else os.path.join(LOG_DIR, "ocr-chunking")
    os.makedirs(target, exist_ok=True)
    return target


def _log_base_dir() -> str:
    """Returns the root log directory for this service."""
    return _ensure_log_dir()


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_FILENAME_RE = re.compile(r"^[\w\-\.]+$")


def _safe_log_path(base: str, *parts: str) -> str:
    """Joins path parts and ensures the result stays within `base`.

    Raises HTTPException(400) if traversal is detected (e.g. '../../etc').
    """
    joined = os.path.normpath(os.path.join(base, *parts))
    real_base = os.path.realpath(base)
    real_joined = os.path.realpath(joined)
    if not real_joined.startswith(real_base + os.sep) and real_joined != real_base:
        raise HTTPException(status_code=400, detail="Ruta de log invalida.")
    return joined


def _today_log_dir() -> str:
    """Returns today's date subdirectory."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _ensure_log_dir(today)


def write_pipeline_log(
    oid: Optional[int],
    stage: str,
    status: str,
    phases: List[Dict[str, Any]],
    data: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    source: str = "pipeline",
) -> str:
    """Writes one pipeline execution log as a JSONL file. Returns the filename."""
    now = datetime.now(timezone.utc)
    ts_file = now.strftime("%Y-%m-%dT%H-%M-%S")
    oid_part = f"_oid-{oid}" if oid is not None else ""
    filename = f"{source}_{ts_file}{oid_part}.jsonl"
    day_dir = _today_log_dir()
    filepath = os.path.join(day_dir, filename)

    log_entry = {
        "timestamp_utc": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "source": source,
        "stage": stage,
        "status": status,
        "oid": oid,
        "hostname": platform.node(),
        "phases": phases,
        "data": data or {},
        "error": error,
    }

    with _LOG_WRITE_LOCK:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_json_safe(log_entry), ensure_ascii=False) + "\n")

    # Also write to consolidated error log if failed
    if status.upper() in {"FAILED", "ERROR"}:
        error_filename = f"errors_{now.strftime('%Y-%m-%d')}.jsonl"
        error_filepath = os.path.join(day_dir, error_filename)
        with _LOG_WRITE_LOCK:
            with open(error_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_json_safe(log_entry), ensure_ascii=False) + "\n")

    return filename


def purge_old_logs(retention_days: Optional[int] = None) -> Dict[str, Any]:
    """Removes log directories older than retention_days. Returns purge summary."""
    days = retention_days if retention_days is not None else LOG_RETENTION_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    base = _log_base_dir()
    removed_dirs: List[str] = []
    removed_files = 0
    freed_bytes = 0

    try:
        for entry in sorted(os.listdir(base)):
            entry_path = os.path.join(base, entry)
            if not os.path.isdir(entry_path):
                continue
            try:
                dir_date = datetime.strptime(entry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if dir_date < cutoff:
                for f in os.listdir(entry_path):
                    fp = os.path.join(entry_path, f)
                    if os.path.isfile(fp):
                        freed_bytes += os.path.getsize(fp)
                        removed_files += 1
                shutil.rmtree(entry_path, ignore_errors=True)
                removed_dirs.append(entry)
    except FileNotFoundError:
        pass

    return {
        "retention_days": days,
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "removed_directories": removed_dirs,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
    }


def _list_log_days() -> List[Dict[str, Any]]:
    """Returns summary per day directory."""
    base = _log_base_dir()
    days: List[Dict[str, Any]] = []
    try:
        for entry in sorted(os.listdir(base), reverse=True):
            entry_path = os.path.join(base, entry)
            if not os.path.isdir(entry_path):
                continue
            try:
                datetime.strptime(entry, "%Y-%m-%d")
            except ValueError:
                continue
            files = [f for f in os.listdir(entry_path) if f.endswith(".jsonl")]
            total_bytes = sum(os.path.getsize(os.path.join(entry_path, f)) for f in files)
            error_files = [f for f in files if f.startswith("errors_")]
            pipeline_files = [f for f in files if not f.startswith("errors_")]
            days.append({
                "date": entry,
                "total_files": len(files),
                "pipeline_logs": len(pipeline_files),
                "error_logs": len(error_files),
                "total_bytes": total_bytes,
                "total_kb": round(total_bytes / 1024, 2),
            })
    except FileNotFoundError:
        pass
    return days


@app.get("/logs/summary", tags=[TAG_LOGS])
def logs_summary(_: Dict[str, Any] = Depends(require_service_auth)) -> Dict[str, Any]:
    """
    Resumen global de logs: dias disponibles, total de archivos,
    espacio en disco, directorio de almacenamiento.
    """
    days = _list_log_days()
    total_files = sum(d["total_files"] for d in days)
    total_bytes = sum(d["total_bytes"] for d in days)
    total_errors = sum(d["error_logs"] for d in days)
    return {
        "timestamp_utc": utc_now_iso(),
        "log_directory": os.path.abspath(_log_base_dir()),
        "retention_days": LOG_RETENTION_DAYS,
        "total_days": len(days),
        "total_files": total_files,
        "total_error_logs": total_errors,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "platform": platform.system(),
        "days": days[:60],
    }


@app.get("/logs/list", tags=[TAG_LOGS])
def logs_list(
    date_filter: Optional[str] = Query(None, alias="date", description="Fecha YYYY-MM-DD"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filtro: error | pipeline | all"),
    limit: int = Query(100, ge=1, le=1000),
    _: Dict[str, Any] = Depends(require_service_auth),
) -> Dict[str, Any]:
    """
    Lista archivos de log con filtros opcionales por fecha y tipo.
    """
    base = _log_base_dir()
    target_dates: List[str] = []

    if date_filter:
        clean_date = date_filter.strip()
        if not _DATE_RE.match(clean_date):
            raise HTTPException(status_code=400, detail="Formato de fecha invalido. Use YYYY-MM-DD.")
        target_dates = [clean_date]
    else:
        try:
            target_dates = sorted(
                [e for e in os.listdir(base) if _DATE_RE.match(e) and os.path.isdir(_safe_log_path(base, e))],
                reverse=True,
            )[:30]
        except FileNotFoundError:
            target_dates = []

    results: List[Dict[str, Any]] = []
    for day in target_dates:
        day_path = _safe_log_path(base, day)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path), reverse=True):
            if not fname.endswith(".jsonl") or not _SAFE_FILENAME_RE.match(fname):
                continue
            if status_filter:
                sf = status_filter.strip().lower()
                if sf == "error" and not fname.startswith("errors_"):
                    continue
                if sf == "pipeline" and fname.startswith("errors_"):
                    continue
            fpath = _safe_log_path(base, day, fname)
            results.append({
                "date": day,
                "filename": fname,
                "size_bytes": os.path.getsize(fpath),
                "size_kb": round(os.path.getsize(fpath) / 1024, 2),
            })
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return {
        "timestamp_utc": utc_now_iso(),
        "filters": {"date": date_filter, "status": status_filter},
        "count": len(results),
        "files": results,
    }


@app.get("/logs/detail/{filename}", tags=[TAG_LOGS])
def logs_detail(
    filename: str,
    date_filter: Optional[str] = Query(None, alias="date", description="Fecha YYYY-MM-DD del log"),
    _: Dict[str, Any] = Depends(require_service_auth),
) -> Dict[str, Any]:
    """
    Retorna el contenido de un archivo de log especifico.
    Si no se pasa date, busca en los ultimos 7 dias.
    """
    base = _log_base_dir()
    clean_name = os.path.basename(filename)
    if not clean_name or not _SAFE_FILENAME_RE.match(clean_name) or not clean_name.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Nombre de archivo de log invalido.")

    search_dates: List[str] = []
    if date_filter:
        clean_date = date_filter.strip()
        if not _DATE_RE.match(clean_date):
            raise HTTPException(status_code=400, detail="Formato de fecha invalido. Use YYYY-MM-DD.")
        search_dates = [clean_date]
    else:
        today = datetime.now(timezone.utc)
        search_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    for day in search_dates:
        fpath = _safe_log_path(base, day, clean_name)
        if os.path.isfile(fpath):
            entries: List[Dict[str, Any]] = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            entries.append({"raw": line})
            return {
                "timestamp_utc": utc_now_iso(),
                "filename": clean_name,
                "date": day,
                "size_bytes": os.path.getsize(fpath),
                "entries_count": len(entries),
                "entries": entries,
            }

    raise HTTPException(status_code=404, detail=f"Log '{clean_name}' no encontrado.")


@app.post("/logs/purge", tags=[TAG_LOGS])
def logs_purge(
    retention_days: Optional[int] = Query(None, description="Dias a retener (default: OCR_LOG_RETENTION_DAYS)"),
    _: Dict[str, Any] = Depends(require_service_auth),
) -> Dict[str, Any]:
    """
    Purga logs mas antiguos que retention_days.
    """
    result = purge_old_logs(retention_days)
    result["timestamp_utc"] = utc_now_iso()
    return result


# ---------------------------------------------------------------------------
# Integration test: upload file and validate full pipeline without DB
# ---------------------------------------------------------------------------

@app.post("/validate-pipeline/upload", tags=[TAG_VALIDATION])
async def validate_pipeline_upload(
    file: UploadFile = File(..., description="Archivo PDF, DOCX, imagen, etc. para validar el pipeline completo."),
    chunking_strategy: str = Query("semantic", description="semantic | simple"),
    embedding_model: str = Query(DEFAULT_EMBEDDING_MODEL, description="Modelo de embeddings a usar"),
    max_chunks: int = Query(0, ge=0, description="Limite de chunks (0 = sin limite)"),
    _: Dict[str, Any] = Depends(require_service_auth),
) -> Dict[str, Any]:
    """
    Validacion integral del pipeline sin persistencia en BD.
    Sube un archivo y ejecuta: deteccion MIME -> OCR/Docling -> limpieza ->
    chunking -> embeddings GPU. Retorna reporte detallado con tiempos por fase,
    calidad OCR, preview de texto, chunks y dimensiones de vectores.

    Ideal para validar que la cadena GPU+librerias funciona de punta a punta
    antes de procesar documentos reales.
    """
    started = time.monotonic()
    recorder = PhaseRecorder()
    result_data: Dict[str, Any] = {"source": "upload_validation"}
    log_filename: Optional[str] = None

    try:
        # --- Phase 1: Read uploaded file ---
        phase_start = time.monotonic()
        file_bytes = await file.read()
        original_filename = safe_str(file.filename, "uploaded_file")
        file_size = len(file_bytes)
        if file_size == 0:
            raise PipelineError("UPLOAD", "EMPTY_FILE", "El archivo subido esta vacio.")
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()

        mime_type = normalize_mime_type(file.content_type)
        if not mime_type:
            mime_type = normalize_mime_type(infer_mime_type_from_file_name(original_filename))
        if not mime_type:
            mime_type = "application/pdf"

        recorder.push("UPLOAD", "OK", "Archivo recibido.", {
            "filename": original_filename,
            "size_bytes": file_size,
            "mime_type": mime_type,
            "sha256": file_sha256,
            "elapsed_ms": int((time.monotonic() - phase_start) * 1000),
        })

        # --- Phase 2: MIME validation ---
        if not is_supported_docling_mime(mime_type):
            raise PipelineError(
                "MIME_VALIDATION", "UNSUPPORTED_MIME_TYPE",
                f"Mime type '{mime_type}' no soportado.",
                {"supported": sorted(SUPPORTED_DOCLING_MIME_TYPES)},
                retryable=False,
            )
        recorder.push("MIME_VALIDATION", "OK", "Mime type soportado.", {"mime_type": mime_type})

        is_pdf = mime_type == "application/pdf"
        extraction_opts = ExtractionOptions()

        # --- Phase 3: Probe (PDF only) ---
        probe: Dict[str, Any] = {}
        engine = "docling"
        if is_pdf:
            phase_start = time.monotonic()
            probe = probe_pdf_extractability(file_bytes, DEFAULT_PROBE_MAX_PAGES)
            engine = resolve_extraction_engine(extraction_opts, probe)
            recorder.push("PROBE", "OK", "Probe completado.", {
                **probe,
                "engine_selected": engine,
                "elapsed_ms": int((time.monotonic() - phase_start) * 1000),
            })
        else:
            recorder.push("PROBE", "SKIPPED", "Probe solo aplica a PDF.")

        # --- Phase 4: Text extraction ---
        phase_start = time.monotonic()
        extraction_meta: Dict[str, Any] = {"engine": engine}
        if is_pdf and engine == "pymupdf":
            extracted_text, extracted_pages = extract_text_pymupdf(file_bytes)
            extraction_meta["ocr_quality"] = safe_float(probe.get("extractable_confidence"), None)
        else:
            extracted_text, extracted_pages, docling_meta = extract_text_docling(
                file_bytes, extraction_opts, mime_type=mime_type, file_name=original_filename,
            )
            extraction_meta.update(docling_meta)

        if not extracted_text.strip():
            raise PipelineError("TEXT_EXTRACTION", "EMPTY_TEXT", "No se extrajo texto del archivo.")

        extraction_meta.update({
            "chars": len(extracted_text),
            "pages": extracted_pages,
            "elapsed_ms": int((time.monotonic() - phase_start) * 1000),
        })
        recorder.push("TEXT_EXTRACTION", "OK", "Extraccion completada.", extraction_meta)

        # --- Phase 5: Cleaning ---
        phase_start = time.monotonic()
        cleaning_opts = CleaningOptions()
        cleaned_text, cleaning_meta = clean_text(extracted_text, cleaning_opts)
        text_for_chunking = cleaned_text if cleaning_opts.enabled else extracted_text
        if not text_for_chunking.strip():
            raise PipelineError("TEXT_CLEANING", "EMPTY_AFTER_CLEANING", "Texto vacio tras limpieza.")
        cleaning_meta["elapsed_ms"] = int((time.monotonic() - phase_start) * 1000)
        recorder.push("TEXT_CLEANING", "OK", "Limpieza completada.", cleaning_meta)

        # --- Phase 6: Chunking ---
        phase_start = time.monotonic()
        chunking_opts = ChunkingOptions(strategy=chunking_strategy, max_chunks=max_chunks)
        if chunking_strategy == "semantic":
            tokenizer_for_chunk: Optional[Any] = None
            try:
                tokenizer_for_chunk = load_tokenizer(embedding_model)
            except Exception:
                pass
            semantic_max_tokens = max(256, derive_embedding_max_length(chunking_opts))
            try:
                chunks = semantic_chunk_text(text_for_chunking, tokenizer_for_chunk, embedding_model, semantic_max_tokens)
            except Exception:
                chunks = simple_chunk_text(text_for_chunking, chunking_opts.simple_chunk_size, chunking_opts.simple_chunk_overlap)
                chunking_strategy = "simple_fallback"
        else:
            chunks = simple_chunk_text(text_for_chunking, chunking_opts.simple_chunk_size, chunking_opts.simple_chunk_overlap)

        if max_chunks > 0:
            chunks = chunks[:max_chunks]
        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            raise PipelineError("CHUNKING", "NO_CHUNKS", "No se generaron chunks.")

        chunking_elapsed = int((time.monotonic() - phase_start) * 1000)
        recorder.push("CHUNKING", "OK", "Chunking completado.", {
            "strategy": chunking_strategy,
            "chunks_count": len(chunks),
            "avg_chunk_chars": round(sum(len(c) for c in chunks) / len(chunks), 1),
            "elapsed_ms": chunking_elapsed,
        })

        # --- Phase 7: Embeddings (GPU) ---
        phase_start = time.monotonic()
        tokenizer, model, device = load_embedding_model(embedding_model)
        derived_max_length = derive_embedding_max_length(chunking_opts)
        vectors, tokens_per_chunk = embed_chunks(
            chunks=chunks, tokenizer=tokenizer, model=model,
            device=device, max_length=derived_max_length, batch_size=8,
        )
        embedding_elapsed = int((time.monotonic() - phase_start) * 1000)
        vector_dim = len(vectors[0]) if vectors else 0

        recorder.push("EMBEDDINGS", "OK", "Embeddings generados en GPU.", {
            "model": embedding_model,
            "device": device,
            "vectors_count": len(vectors),
            "vector_dimension": vector_dim,
            "total_tokens": sum(tokens_per_chunk),
            "elapsed_ms": embedding_elapsed,
        })

        # --- Build result ---
        total_elapsed = int((time.monotonic() - started) * 1000)
        result_data.update({
            "status": "COMPLETED",
            "filename": original_filename,
            "file_size_bytes": file_size,
            "mime_type": mime_type,
            "sha256": file_sha256,
            "engine_used": engine,
            "ocr_quality": safe_float(extraction_meta.get("ocr_quality"), None),
            "text_chars": len(text_for_chunking),
            "text_words": count_words(text_for_chunking),
            "text_preview": text_for_chunking[:1000],
            "pages": extracted_pages,
            "chunks_count": len(chunks),
            "chunks_preview": [c[:200] for c in chunks[:5]],
            "vectors_count": len(vectors),
            "vector_dimension": vector_dim,
            "embedding_model": embedding_model,
            "gpu_device": device,
            "gpu_metrics": gpu_metrics(),
            "total_elapsed_ms": total_elapsed,
            "timing": {
                "extraction_ms": extraction_meta.get("elapsed_ms"),
                "cleaning_ms": cleaning_meta.get("elapsed_ms"),
                "chunking_ms": chunking_elapsed,
                "embedding_ms": embedding_elapsed,
                "total_ms": total_elapsed,
            },
            "persisted_to_db": False,
        })

        log_filename = write_pipeline_log(
            oid=None, stage="validate_upload", status="COMPLETED",
            phases=recorder.as_list(), data=result_data, source="validation",
        )

        return {
            "timestamp_utc": utc_now_iso(),
            "status": "ok",
            "message": "Pipeline integral validado exitosamente.",
            "log_file": log_filename,
            "phases": recorder.as_list(),
            "result": result_data,
        }

    except PipelineError as exc:
        recorder.push(exc.phase, "ERROR", exc.message, exc.details)
        error_data = exc.to_dict()
        total_elapsed = int((time.monotonic() - started) * 1000)
        result_data.update({"status": "FAILED", "total_elapsed_ms": total_elapsed})
        log_filename = write_pipeline_log(
            oid=None, stage="validate_upload", status="FAILED",
            phases=recorder.as_list(), data=result_data, error=error_data, source="validation",
        )
        raise HTTPException(status_code=422, detail=to_json_safe({
            "status": "FAILED",
            "message": exc.message,
            "error": error_data,
            "phases": recorder.as_list(),
            "log_file": log_filename,
            "total_elapsed_ms": total_elapsed,
        }))
    except HTTPException:
        raise
    except Exception as exc:
        recorder.push("UNHANDLED", "ERROR", str(exc), {"traceback": traceback.format_exc()})
        total_elapsed = int((time.monotonic() - started) * 1000)
        result_data.update({"status": "FAILED", "total_elapsed_ms": total_elapsed})
        log_filename = write_pipeline_log(
            oid=None, stage="validate_upload", status="FAILED",
            phases=recorder.as_list(), error={"message": str(exc)}, source="validation",
        )
        raise HTTPException(status_code=500, detail=to_json_safe({
            "status": "FAILED",
            "message": f"{type(exc).__name__}: {str(exc)}",
            "phases": recorder.as_list(),
            "log_file": log_filename,
            "total_elapsed_ms": total_elapsed,
        }))


def _raise_pipeline_error(detail: Dict[str, Any]) -> None:
    """Lanza HTTP 422 con detalle estructurado para errores de pipeline (no auth)."""
    raise HTTPException(status_code=422, detail=to_json_safe(detail))


def _raise_forbidden(detail: Dict[str, Any]) -> None:
    """Lanza HTTP 403 con detalle estructurado y serializable."""
    raise HTTPException(status_code=403, detail=to_json_safe(detail))


def _run_single_stage_or_403(payload: Dict[str, Any], stage: str) -> OCRChunkingResponse:
    """Ejecuta una solicitud single-stage y eleva 422 con detalle completo si falla."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingRequest(**input_payload)
    except Exception as exc:
        _raise_pipeline_error(
            {
                "status": "FAILED",
                "exitoso": False,
                "stage": stage,
                "message": "Payload invalido.",
                "error": {
                    "phase": "REQUEST_PARSING",
                    "code": "INVALID_PAYLOAD",
                    "message": f"{type(exc).__name__}: {str(exc)}",
                    "details": {"traceback": traceback.format_exc()},
                },
                "input": to_json_safe(input_payload),
            }
        )

    response = process_request(request, stage=stage)
    if not response.exitoso:
        _raise_pipeline_error(
            {
                "status": response.status,
                "exitoso": response.exitoso,
                "stage": stage,
                "message": response.message,
                "error": to_json_safe(response.error or {}),
                "phases": to_json_safe(response.phases),
                "data": to_json_safe(response.data),
            }
        )
    return response


def _run_batch_stage_or_403(payload: Dict[str, Any], stage: str) -> OCRChunkingBatchResponse:
    """Ejecuta una solicitud batch y eleva 403 con detalle completo si falla/parcial."""
    input_payload = payload.get("input", payload)
    try:
        request = OCRChunkingBatchRequest(**input_payload)
    except Exception as exc:
        _raise_pipeline_error(
            {
                "status": "FAILED",
                "exitoso": False,
                "stage": stage,
                "message": "Payload batch invalido.",
                "error": {
                    "phase": "REQUEST_PARSING",
                    "code": "INVALID_BATCH_PAYLOAD",
                    "message": f"{type(exc).__name__}: {str(exc)}",
                    "details": {"traceback": traceback.format_exc()},
                },
                "input": to_json_safe(input_payload),
            }
        )

    response = process_batch(request, stage=stage)
    if not response.exitoso:
        _raise_pipeline_error(
            {
                "status": response.status,
                "exitoso": response.exitoso,
                "stage": stage,
                "message": response.message,
                "total": response.total,
                "completados": response.completados,
                "fallidos": response.fallidos,
                "resultados": to_json_safe([pydantic_model_dump(r) for r in response.resultados]),
            }
        )
    return response


@app.post("/ocr-docling/process", response_model=OCRChunkingResponse, tags=[TAG_OCR])
def ocr_docling_process(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingResponse:
    """Ejecuta solo OCR + limpieza de texto."""
    return _run_single_stage_or_403(payload, stage="ocr")


@app.post("/ocr-docling/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_OCR])
def ocr_docling_batch(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingBatchResponse:
    """Ejecuta OCR + limpieza para varios documentos."""
    return _run_batch_stage_or_403(payload, stage="ocr")


@app.post("/chunking-docling/process", response_model=OCRChunkingResponse, tags=[TAG_CHUNKING])
def chunking_docling_process(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingResponse:
    """Ejecuta OCR + limpieza + chunking."""
    return _run_single_stage_or_403(payload, stage="chunking")


@app.post("/chunking-docling/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_CHUNKING])
def chunking_docling_batch(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingBatchResponse:
    """Ejecuta OCR + limpieza + chunking para varios documentos."""
    return _run_batch_stage_or_403(payload, stage="chunking")


@app.post("/embedding-generation/process", response_model=OCRChunkingResponse, tags=[TAG_EMBEDDING])
def embedding_generation_process(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingResponse:
    """Ejecuta OCR + limpieza + chunking + generacion de embeddings."""
    return _run_single_stage_or_403(payload, stage="embedding")


@app.post("/embedding-generation/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_EMBEDDING])
def embedding_generation_batch(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingBatchResponse:
    """Ejecuta generacion de embeddings para varios documentos."""
    return _run_batch_stage_or_403(payload, stage="embedding")


@app.post("/PipelineOCR/process", response_model=OCRChunkingResponse, tags=[TAG_PIPELINE])
def pipeline_ocr_process(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingResponse:
    """Orquesta todo el flujo: OCR, limpieza, chunking, embeddings e insercion."""
    return _run_single_stage_or_403(payload, stage="pipeline")


@app.post("/PipelineOCR/process-batch", response_model=OCRChunkingBatchResponse, tags=[TAG_PIPELINE])
def pipeline_ocr_batch(
    payload: Dict[str, Any],
    _: Dict[str, Any] = Depends(require_service_auth),
) -> OCRChunkingBatchResponse:
    """Orquesta flujo completo para varios documentos."""
    return _run_batch_stage_or_403(payload, stage="pipeline")


def run_mock_local_demo(args: argparse.Namespace) -> None:
    """Runs local mock example from CLI and prints JSON response."""
    request = OCRChunkingRequest(
        oid=int(args.mock_oid),
        mock=MockOptions(
            enabled=True,
            fail_phase=args.mock_fail_phase,
            latency_ms=int(args.mock_latency_ms),
            without_db=True,
        ),
    )
    result = process_request(request)
    print(pydantic_model_dump_json(result, indent=2))


def parse_args() -> argparse.Namespace:
    """Parses CLI args."""
    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--mock-local", action="store_true", help="Runs mock pipeline once and exits.")
    parser.add_argument("--mock-oid", type=int, default=2299268)
    parser.add_argument("--mock-fail-phase", default=None)
    parser.add_argument("--mock-latency-ms", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.mock_local:
        run_mock_local_demo(args)
        return
    uvicorn.run(app, host=args.host, port=args.port, reload=bool(args.reload))


if __name__ == "__main__":
    main()