# Brain VT – OCR Chunking Embeddings Service

Servicio OpenAPI (FastAPI) que orquesta **OCR, chunking semántico y generación de embeddings** para documentos almacenados como Large Objects en PostgreSQL. Diseñado para ejecutarse con aceleración GPU (NVIDIA).

---

## Flujo de trabajo y gobernanza

| Rama | Propósito |
|---|---|
| `main` | Rama estable y de release — solo recibe cambios desde `develop` vía PR |
| `develop` | Rama de integración de trabajo diario |

### Convención de ramas

- `feature/<alcance>` para nuevas capacidades
- `fix/<alcance>` para correcciones normales  
- `hotfix/<alcance>` para correcciones urgentes con destino directo a main

### Flujo recomendado

1. Crear rama desde `develop`.
2. Abrir PR hacia `develop` para cambios normales.
3. Cuando `develop` esté estable, promover hacia `main` con PR.
4. Ramas temporales se borran automáticamente tras merge.

---

## Tabla de contenidos

1. [Arquitectura general](#arquitectura-general)
2. [Requisitos de hardware](#requisitos-de-hardware)
3. [Requisitos de software](#requisitos-de-software)
4. [Instalación paso a paso](#instalación-paso-a-paso)
5. [Variables de entorno](#variables-de-entorno)
6. [Arranque del servicio](#arranque-del-servicio)
7. [Validación post-despliegue](#validación-post-despliegue)
8. [Catálogo de endpoints](#catálogo-de-endpoints)
9. [Autenticación JWT](#autenticación-jwt)
10. [Pipeline de procesamiento](#pipeline-de-procesamiento)
11. [Sistema de logs](#sistema-de-logs)
12. [Compatibilidad GPU](#compatibilidad-gpu)
13. [Modelo de embeddings](#modelo-de-embeddings)
14. [Formatos de archivo soportados](#formatos-de-archivo-soportados)
15. [Ejemplo de request/response](#ejemplo-de-requestresponse)
16. [Troubleshooting](#troubleshooting)

---

## Arquitectura general

```
┌──────────────┐     HTTP/JSON      ┌─────────────────────────────────────────┐
│  Caller      │ ─────────────────► │  ocr_chunking.py (FastAPI + Uvicorn)    │
│  (Airflows)  │ ◄───────────────── │                                         │
└──────────────┘                    │  ┌──────────┐ ┌────────┐ ┌───────────┐  │
                                    │  │ Docling   │ │PyMuPDF │ │ RapidOCR  │  │
                                    │  │ OCR/Parse │ │FastPath│ │ ONNX+GPU  │  │
                                    │  └────┬─────┘ └───┬────┘ └─────┬─────┘  │
                                    │       └───────────┴────────────┘        │
                                    │                  │                       │
                                    │       ┌──────────▼──────────┐           │
                                    │       │ HybridChunker       │           │
                                    │       │ (docling-core)      │           │
                                    │       └──────────┬──────────┘           │
                                    │                  │                       │
                                    │       ┌──────────▼──────────┐           │
                                    │       │ Embeddings GPU      │           │
                                    │       │ multilingual-e5     │           │
                                    │       │ (torch fp16+AMP)    │           │
                                    │       └──────────┬──────────┘           │
                                    │                  │                       │
                                    └──────────────────┼───────────────────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   PostgreSQL     │
                                              │   (pgvector)     │
                                              └─────────────────┘
```

**Archivo principal único**: `ocr_chunking.py` (~4400 líneas). No hay módulos adicionales.

---

## Requisitos de hardware

### Producción (RTX A6000 Ampere)

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU | 1× NVIDIA RTX A6000 (48 GB VRAM) | 3× RTX A6000 |
| CPU | 8 cores | 16+ cores |
| RAM | 32 GB | 64 GB |
| Disco | 50 GB (modelos + logs) | 100 GB |
| Arquitectura GPU | Ampere (sm_86, compute capability 8.6) | — |

### Desarrollo (RTX 5090 Blackwell)

| Componente | Valor |
|---|---|
| GPU | NVIDIA RTX 5090 Laptop (32 GB VRAM) |
| Arquitectura GPU | Blackwell (sm_120, compute capability 12.0) |

> **Importante**: La versión de PyTorch (cu126 vs cu130) cambia según la GPU. Ver sección [Compatibilidad GPU](#compatibilidad-gpu).

---

## Requisitos de software

| Software | Versión |
|---|---|
| Python | 3.10+ (probado con 3.14.3) |
| NVIDIA Driver | ≥ 535.x (Ampere), ≥ 595.x (Blackwell) |
| CUDA Toolkit | Incluido en PyTorch (no requiere instalación separada) |
| PostgreSQL | 14+ con extensión `pgvector` |
| Sistema operativo | Linux (producción) / Windows 11 (desarrollo) |

---

## Instalación paso a paso

### 1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux
# venv\Scripts\activate          # Windows
```

### 2. Instalar PyTorch + torchvision (SEGÚN LA GPU)

**Para RTX A6000 (Ampere, sm_86) — PRODUCCIÓN:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Para RTX 5090 (Blackwell, sm_120) — DESARROLLO:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> **Este paso DEBE hacerse ANTES de instalar requirements.txt.** PyTorch no está en requirements.txt porque su versión depende de la GPU del host.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Arch list:', torch.cuda.get_arch_list())
t = torch.randn(3,3, device='cuda')
print('Tensor test OK:', t.device)
"
```

**Salida esperada para RTX A6000:**
```
torch: 2.10.0+cu126
CUDA: True
GPU: NVIDIA RTX A6000
Arch list: ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
Tensor test OK: cuda:0
```

> `sm_86` **DEBE** aparecer en `Arch list`. Si no aparece, la GPU fallará en runtime.

### 5. Verificar ONNX Runtime GPU

```bash
python -c "
import onnxruntime as ort
print('onnxruntime:', ort.__version__)
print('Providers:', ort.get_available_providers())
"
```

**Salida esperada:**
```
onnxruntime: 1.24.4
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

> `CUDAExecutionProvider` **DEBE** aparecer. Lo usa RapidOCR para OCR con GPU.

---

## Variables de entorno

### Base de datos (obligatorias en producción)

| Variable | Default | Descripción |
|---|---|---|
| `OCR_DB_HOST` | `localhost` | Host PostgreSQL |
| `OCR_DB_PORT` | `5432` | Puerto PostgreSQL |
| `OCR_DB_NAME` | `niledb` | Nombre de la base de datos |
| `OCR_DB_USER` | `postgres` | Usuario de conexión |
| `OCR_DB_PASSWORD` | `plexia` | Contraseña |

### Autenticación JWT

| Variable | Default | Descripción |
|---|---|---|
| `OCR_AUTH_ENABLED` | `true` | Habilitar validación JWT. Poner `false` solo para testing local |
| `OCR_JWT_ISSUER` | `https://anh-pro.flows.ninja/auth/realms/airflows` | Issuer esperado en el token |
| `OCR_JWKS_URL` | `{issuer}/protocol/openid-connect/certs` | URL del JWKS de Keycloak |
| `OCR_JWT_AUDIENCE` | _(vacío)_ | Si se configura, valida claim `aud` |
| `OCR_JWT_EXPECTED_AZP` | `ocr-chunking-embeddings` | Valida claim `azp` |
| `OCR_JWT_EXPECTED_CLIENT_ID` | `ocr-chunking-embeddings` | Valida claim `clientId` |
| `OCR_JWT_ALGORITHMS` | `RS256` | Algoritmos aceptados (separados por coma) |

### Logs

| Variable | Default | Descripción |
|---|---|---|
| `OCR_LOG_DIR` | `{tempdir}/ocr-chunking-logs` | Directorio raíz de logs JSONL |
| `OCR_LOG_RETENTION_DAYS` | `30` | Días a retener antes de purgar |
| `OCR_LOG_LEVEL` | `INFO` | Nivel de logging estándar (stdout) |

### Ejemplo de `.env` para producción

```env
# PostgreSQL
OCR_DB_HOST=10.0.1.50
OCR_DB_PORT=5432
OCR_DB_NAME=niledb
OCR_DB_USER=ocr_service
OCR_DB_PASSWORD=<password_seguro>

# JWT / Keycloak
OCR_AUTH_ENABLED=true
OCR_JWT_ISSUER=https://anh-pro.flows.ninja/auth/realms/airflows
OCR_JWT_EXPECTED_AZP=ocr-chunking-embeddings

# Logs
OCR_LOG_DIR=/var/log/ocr-chunking
OCR_LOG_RETENTION_DAYS=60
OCR_LOG_LEVEL=INFO
```

---

## Arranque del servicio

### Directo (desarrollo y testing)

```bash
python ocr_chunking.py --host 0.0.0.0 --port 8000
```

### Con Uvicorn (producción)

```bash
uvicorn ocr_chunking:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Workers = 1** es obligatorio. El servicio mantiene modelos GPU en memoria (cache singleton) que no se comparten entre procesos. Usar más de 1 worker duplicaría el consumo de VRAM y causaría errores OOM.

### Con systemd (Linux)

```ini
# /etc/systemd/system/ocr-chunking.service
[Unit]
Description=Brain VT OCR Chunking Embeddings Service
After=network.target postgresql.service

[Service]
Type=simple
User=ocr-service
WorkingDirectory=/opt/brain-vt-ocr-chunking
EnvironmentFile=/opt/brain-vt-ocr-chunking/.env
ExecStart=/opt/brain-vt-ocr-chunking/venv/bin/uvicorn ocr_chunking:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# GPU access
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-chunking
sudo systemctl start ocr-chunking
sudo journalctl -u ocr-chunking -f    # ver logs
```

### Primer arranque — descarga de modelos

En el **primer request** que use embeddings, el servicio descarga el modelo de HuggingFace:

```
intfloat/multilingual-e5-large-instruct (~2.2 GB)
```

Esto tarda 2-5 minutos según el ancho de banda. Los requests posteriores usan la cache local en `~/.cache/huggingface/`.

**Para pre-descargar** el modelo antes de recibir tráfico:

```bash
python -c "
from transformers import AutoModel, AutoTokenizer
name = 'intfloat/multilingual-e5-large-instruct'
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print('Modelo descargado OK')
"
```

---

## Validación post-despliegue

Después de desplegar, ejecutar estos endpoints **en orden** para verificar que todo funciona:

### Paso 1: Health check básico

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/health
```

Respuesta esperada: `{"status": "ok", ...}`

### Paso 2: Validar GPU

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/validate-gpu
```

Verificar en la respuesta:
- `status` = `"ok"`
- `devices[0].name` = `"NVIDIA RTX A6000"`
- `devices[0].architecture.sm_code` = `"sm_86"`
- `devices[0].compatible_with_torch` = `true`
- `onnxruntime.gpu_enabled` = `true`

### Paso 3: Stress test CUDA

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/validate-cuda-stress
```

Verificar: `status` = `"ok"` y cada device con `status` = `"ok"`.

### Paso 4: Validar librerías

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/validate-libraries
```

Verificar: `missing` = `[]` (array vacío).

### Paso 5: Validar BD

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/validate-db
```

Verificar: `status` = `"ok"`.

### Paso 6: Resumen ejecutivo

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/validate-environment
```

Verificar: `ready_for_production` = `true`. Si es `false`, el campo `checks` indica qué falló.

### Paso 7: Test integral con archivo

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@documento_prueba.pdf" \
  http://localhost:8000/validate-pipeline/upload
```

Este endpoint ejecuta el pipeline completo (OCR → limpieza → chunking → embeddings GPU) **sin escribir en BD**. La respuesta incluye tiempos por fase, preview del texto, chunks y vectores generados. Si retorna `status: "ok"`, el servicio está 100% operativo.

---

## Catálogo de endpoints

### Helpers

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Health check con métricas GPU |
| GET | `/example-request` | Ejemplo de payload para los endpoints de procesamiento |
| GET | `/validate-db` | Valida conexión a PostgreSQL |

### Infraestructura

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/validate-gpu` | Diagnóstico completo de GPU: arquitectura, VRAM, compatibilidad torch/ONNX |
| GET | `/validate-libraries` | Versiones de todas las librerías críticas |
| GET | `/validate-cuda-stress` | Test real de ejecución CUDA (matmul) en cada GPU |
| GET | `/validate-environment` | Resumen ejecutivo: ¿está listo para producción? |

### Validación integral

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/validate-pipeline/upload` | Sube un archivo y ejecuta pipeline completo sin BD |

### Logs

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/logs/summary` | Resumen global: días, archivos, espacio en disco |
| GET | `/logs/list?date=YYYY-MM-DD&status=error` | Lista archivos de log con filtros |
| GET | `/logs/detail/{filename}?date=YYYY-MM-DD` | Contenido de un log específico |
| POST | `/logs/purge?retention_days=30` | Purga logs antiguos |

### Procesamiento (endpoints funcionales)

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/ocr-docling/process` | Solo OCR: extrae texto del documento |
| POST | `/ocr-docling/process-batch` | OCR en lote |
| POST | `/chunking-docling/process` | OCR + chunking |
| POST | `/chunking-docling/process-batch` | OCR + chunking en lote |
| POST | `/embedding-generation/process` | OCR + chunking + embeddings |
| POST | `/embedding-generation/process-batch` | OCR + chunking + embeddings en lote |
| POST | `/PipelineOCR/process` | Pipeline completo: OCR + chunking + embeddings + persistencia BD |
| POST | `/PipelineOCR/process-batch` | Pipeline completo en lote |

Todos los endpoints de procesamiento reciben el mismo modelo de request (`OCRChunkingRequest`) y difieren en las fases que ejecutan.

---

## Autenticación JWT

Todos los endpoints requieren header `Authorization: Bearer <token>` validado contra Keycloak JWKS.

**Claims validados:**
- `iss` — issuer obligatorio
- `exp` — expiración obligatoria
- `iat` — issued at obligatorio
- `aud` — solo si se configura `OCR_JWT_AUDIENCE`
- `azp` — si se configura `OCR_JWT_EXPECTED_AZP`
- `clientId` — si se configura `OCR_JWT_EXPECTED_CLIENT_ID`

Para **deshabilitar auth** en testing: `OCR_AUTH_ENABLED=false`.

---

## Pipeline de procesamiento

```
Request (oid) → Resolver documentoId → Leer Large Object → Selección de páginas
    → Probe extractabilidad → Extracción texto (Docling OCR / PyMuPDF fast path)
    → Limpieza determinista → Chunking (semántico / simple)
    → Embeddings GPU (fp16 + AMP) → Persistencia pgvector → Actualizar documento
```

### Motor de extracción

El servicio elige automáticamente (`engine: "auto"`):

- **PyMuPDF fast path**: si el PDF tiene texto embebido con confianza ≥ 0.85. Más rápido, no usa GPU.
- **Docling OCR**: si el PDF es escaneado/imagen o confianza < 0.85. Usa RapidOCR con GPU via ONNX Runtime.

### Optimizaciones GPU

- **Modelo en float16**: reduce VRAM a la mitad (~1.1 GB vs ~2.2 GB para e5-large)
- **`torch.inference_mode()`**: menor overhead que `no_grad()`
- **`torch.amp.autocast("cuda")`**: mixed precision automática en inferencia
- **ONNX Runtime GPU**: RapidOCR usa `CUDAExecutionProvider` directamente, sin pasar por PyTorch

---

## Sistema de logs

Cada ejecución del pipeline escribe un archivo `.jsonl` estructurado:

```
{LOG_DIR}/ocr-chunking/
  2026-03-21/
    pipeline_2026-03-21T10-30-00_oid-2299268.jsonl    ← ejecución exitosa
    pipeline_2026-03-21T10-35-12_oid-4455667.jsonl    ← ejecución exitosa
    validation_2026-03-21T11-00-00.jsonl               ← test de validación
    errors_2026-03-21.jsonl                             ← consolidado de errores del día
```

**Directorio por defecto:**
- Linux: `/tmp/ocr-chunking-logs/ocr-chunking/`
- Windows: `C:\Users\<user>\AppData\Local\Temp\ocr-chunking-logs\ocr-chunking\`

**Recomendación producción**: configurar `OCR_LOG_DIR=/var/log/ocr-chunking` con permisos del usuario del servicio.

Los logs **no reemplazan** stdout/stderr. El servicio escribe a ambos: logging estándar para Docker/journald y archivos JSONL para consulta vía API.

---

## Compatibilidad GPU

| GPU | Familia | Compute Capability | PyTorch Index URL | Build |
|---|---|---|---|---|
| RTX A6000 | Ampere | 8.6 (sm_86) | `https://download.pytorch.org/whl/cu126` | `torch==2.10.0+cu126` |
| RTX 3090 | Ampere | 8.6 (sm_86) | `https://download.pytorch.org/whl/cu126` | `torch==2.10.0+cu126` |
| RTX 4090 | Ada Lovelace | 8.9 (sm_89) | `https://download.pytorch.org/whl/cu126` | `torch==2.10.0+cu126` |
| RTX 5090 | Blackwell | 12.0 (sm_120) | `https://download.pytorch.org/whl/cu130` | `torch==2.10.0+cu130` |
| A100 | Ampere | 8.0 (sm_80) | `https://download.pytorch.org/whl/cu126` | `torch==2.10.0+cu126` |
| H100 | Hopper | 9.0 (sm_90) | `https://download.pytorch.org/whl/cu126` | `torch==2.10.0+cu126` |

> **Regla**: si la GPU es Blackwell (RTX 50xx), usar `cu130`. Para todo lo demás, usar `cu126`.

El endpoint `GET /validate-gpu` verifica esta compatibilidad automáticamente.

---

## Modelo de embeddings

| Propiedad | Valor |
|---|---|
| Nombre | `intfloat/multilingual-e5-large-instruct` |
| Dimensión del vector | 1024 |
| Idiomas | Multilingüe (incluye español) |
| Tamaño en disco | ~2.2 GB |
| VRAM en fp16 | ~1.1 GB |
| Fuente | HuggingFace |
| Cache local | `~/.cache/huggingface/hub/` |

---

## Formatos de archivo soportados

| Formato | MIME Type | Motor |
|---|---|---|
| PDF | `application/pdf` | Docling OCR / PyMuPDF |
| DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | Docling |
| XLSX | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | Docling |
| PPTX | `application/vnd.openxmlformats-officedocument.presentationml.presentation` | Docling |
| Markdown | `text/markdown` | Docling |
| AsciiDoc | `text/asciidoc` | Docling |
| LaTeX | `text/x-tex`, `application/x-latex` | Docling |
| HTML | `text/html`, `application/xhtml+xml` | Docling |
| CSV | `text/csv` | Docling |
| PNG | `image/png` | Docling OCR |
| JPEG | `image/jpeg` | Docling OCR |
| TIFF | `image/tiff` | Docling OCR |
| BMP | `image/bmp` | Docling OCR |
| WebP | `image/webp` | Docling OCR |

---

## Ejemplo de request/response

### Request — Pipeline completo

```bash
curl -X POST http://localhost:8000/PipelineOCR/process \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "oid": 2299268,
      "nombre_documento": "CTO_EyP_LLA_50_2013.pdf",
      "mime_type": "application/pdf",
      "documento_id": 1234,
      "created_by": 1101,
      "extraction": {
        "engine": "auto",
        "page_mode": "head_tail",
        "head_pages": 8,
        "tail_pages": 8
      },
      "chunking": {
        "strategy": "semantic"
      },
      "embedding": {
        "enabled": true,
        "save_to_db": true
      }
    }
  }'
```

### Response exitoso

```json
{
  "status": "COMPLETED",
  "exitoso": true,
  "message": "Proceso 'pipeline' completado exitosamente.",
  "error": null,
  "phases": [
    {"phase": "REQUEST_VALIDATION", "status": "OK", "message": "Request recibida.", ...},
    {"phase": "LOAD_BINARY", "status": "OK", ...},
    {"phase": "PAGE_SELECTION", "status": "OK", ...},
    {"phase": "TEXT_EXTRACTION", "status": "OK", ...},
    {"phase": "TEXT_CLEANING", "status": "OK", ...},
    {"phase": "CHUNKING", "status": "OK", ...},
    {"phase": "EMBEDDINGS", "status": "OK", ...},
    {"phase": "PERSIST", "status": "OK", ...},
    {"phase": "DOCUMENT_FINALIZE", "status": "OK", ...}
  ],
  "data": {
    "oid": 2299268,
    "documento_id_resuelto": 1234,
    "engine_used": "docling",
    "chunks_count": 47,
    "inserted_rows": 47,
    "elapsed_ms": 12450,
    "gpu": {"cuda_available": true, "device_name": "NVIDIA RTX A6000", ...}
  }
}
```

### Response con error

```json
{
  "status": "FAILED",
  "exitoso": false,
  "message": "Proceso fallido.",
  "error": {
    "phase": "TEXT_EXTRACTION",
    "code": "EMPTY_EXTRACTED_TEXT",
    "message": "No se pudo extraer texto del archivo.",
    "details": {"engine": "docling"}
  },
  "phases": [...]
}
```

---

## Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`

**Causa**: PyTorch fue compilado para una arquitectura GPU diferente a la instalada.

**Solución**: Reinstalar PyTorch con el CUDA index correcto para la GPU:

```bash
# Para RTX A6000 (Ampere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Para RTX 5090 (Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

Verificar con `GET /validate-gpu` que `compatible_with_torch` = `true`.

### `ModuleNotFoundError: No module named 'onnxruntime'`

**Causa**: onnxruntime-gpu no instalado. RapidOCR hace fallback a PyTorch para OCR, que puede fallar en GPUs nuevas.

**Solución**:

```bash
pip install onnxruntime-gpu==1.24.4
```

### OOM (Out of Memory) en GPU

**Causa**: Modelo cargado en fp32 o batch_size muy alto.

**Solución**: El servicio ya carga modelos en fp16 por defecto. Si persiste:
- Reducir `embedding.batch_size` en el request (default: 8)
- Verificar que no hay otros procesos usando la GPU: `nvidia-smi`

### Primer request muy lento (~2-5 min)

**Causa normal**: El modelo de embeddings se descarga de HuggingFace la primera vez.

**Solución**: Pre-descargar antes de recibir tráfico:

```bash
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
"
```

### `psycopg2.OperationalError: could not connect to server`

**Causa**: Variables de entorno de BD no configuradas o PostgreSQL no accesible.

**Solución**: Verificar `OCR_DB_HOST`, `OCR_DB_PORT`, `OCR_DB_NAME`, `OCR_DB_USER`, `OCR_DB_PASSWORD`. Probar con `GET /validate-db`.

### `torchvision CUDA version mismatch`

**Causa**: torch y torchvision instalados desde índices CUDA diferentes.

**Solución**: Instalar ambos del mismo índice:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

---

## Versiones de librerías (referencia)

Estas son las versiones pinadas en `requirements.txt` y probadas en conjunto:

| Librería | Versión |
|---|---|
| torch | 2.10.0+cu126 (Ampere) / +cu130 (Blackwell) |
| torchvision | 0.25.0+cu126 / +cu130 |
| docling | 2.81.0 |
| docling-core | 2.70.2 |
| docling-ibm-models | 3.12.0 |
| docling-parse | 5.6.0 |
| rapidocr | 3.7.0 |
| onnxruntime-gpu | 1.24.4 |
| transformers | 4.57.6 |
| accelerate | 1.13.0 |
| huggingface-hub | 0.36.2 |
| fastapi | 0.135.1 |
| uvicorn | 0.42.0 |
| pydantic | 2.12.5 |
| pymupdf | 1.27.2.2 |
| numpy | 2.4.3 |
| Pillow | 12.1.1 |
| psycopg2-binary | 2.9.11 |
| safetensors | 0.7.0 |
| tokenizers | 0.22.2 |
| sentencepiece | 0.2.1 |
| PyJWT | 2.12.1 |

> **Nota**: `transformers` se mantiene en 4.57.6 porque `docling-ibm-models` requiere `transformers<5.0.0` y `huggingface_hub<1`. Esta restricción viene del ecosistema docling.
