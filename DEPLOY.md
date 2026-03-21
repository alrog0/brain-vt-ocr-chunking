# Brain VT - OCR Chunking Embeddings Service
# Guia de Instalacion, Configuracion y Despliegue

## Tabla de contenidos

1. [Descripcion del servicio](#1-descripcion-del-servicio)
2. [Formatos soportados](#2-formatos-soportados)
3. [Requisitos de infraestructura](#3-requisitos-de-infraestructura)
4. [Instalacion paso a paso](#4-instalacion-paso-a-paso)
5. [Variables de entorno](#5-variables-de-entorno)
6. [Arranque del servicio](#6-arranque-del-servicio)
7. [Validacion post-despliegue](#7-validacion-post-despliegue)
8. [Endpoints disponibles](#8-endpoints-disponibles)
9. [Arquitectura GPU y compatibilidad](#9-arquitectura-gpu-y-compatibilidad)
10. [Optimizacion de rendimiento](#10-optimizacion-de-rendimiento)
11. [Logs y observabilidad](#11-logs-y-observabilidad)
12. [Troubleshooting](#12-troubleshooting)
13. [Despliegue con Docker](#13-despliegue-con-docker)

**Python**: 3.10 | 3.11 | 3.12 (recomendado) | 3.13 | 3.14

---

## 1. Descripcion del servicio

Microservicio FastAPI que procesa documentos en multiples formatos mediante:

1. **OCR** - Extraccion de texto con Docling + RapidOCR (GPU accelerado)
2. **Chunking** - Division semantica del texto en fragmentos
3. **Embeddings** - Generacion de vectores con modelos transformer (GPU)

El servicio expone endpoints REST protegidos con JWT y persiste resultados
en PostgreSQL.

---

## 2. Formatos soportados

Basado en Docling 2.81.0 — 17 formatos de entrada.

### Documentos

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| PDF           | `.pdf`               | OCR + layout analysis en GPU             |
| DOCX          | `.docx`              | Microsoft Word Open XML                  |
| XLSX          | `.xlsx`              | Microsoft Excel Open XML                 |
| PPTX          | `.pptx`              | Microsoft PowerPoint Open XML            |

### Texto y markup

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| HTML / XHTML  | `.html`, `.htm`, `.xhtml` | Paginas web                         |
| Markdown      | `.md`, `.markdown`   | Incluye variantes `.qmd`, `.Rmd`         |
| AsciiDoc      | `.adoc`, `.asciidoc` |                                          |
| LaTeX         | `.tex`, `.latex`     |                                          |
| CSV           | `.csv`               | Datos tabulares                          |
| Plain text    | `.txt`, `.text`      |                                          |

### Imagenes (OCR)

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| PNG           | `.png`               | OCR con RapidOCR (GPU)                   |
| JPEG          | `.jpg`, `.jpeg`      | OCR con RapidOCR (GPU)                   |
| TIFF          | `.tif`, `.tiff`      | Comun en documentos escaneados           |
| BMP           | `.bmp`               |                                          |
| WebP          | `.webp`              |                                          |

### Audio y video (requiere `docling[asr]`)

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| WAV           | `.wav`               | Transcripcion automatica (ASR)           |
| MP3           | `.mp3`               | Transcripcion automatica (ASR)           |
| M4A / AAC     | `.m4a`, `.aac`       | Transcripcion automatica (ASR)           |
| OGG           | `.ogg`               | Transcripcion automatica (ASR)           |
| FLAC          | `.flac`              | Transcripcion automatica (ASR)           |
| MP4 / AVI / MOV| `.mp4`, `.avi`, `.mov` | Requiere ffmpeg instalado             |

### Subtitulos

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| WebVTT        | `.vtt`               | Subtitulos / transcripciones             |

### Esquemas XML especializados

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| USPTO XML     | `.xml`               | Patentes (US Patent Office)              |
| JATS XML      | `.xml`               | Articulos academicos                     |
| XBRL XML      | `.xml`               | Reportes financieros                     |
| METS/GBS      | `.xml`               | Google Books metadata                    |

### Re-importacion

| Formato       | Extensiones          | Notas                                    |
|---------------|----------------------|------------------------------------------|
| Docling JSON  | `.json`              | Re-importar documentos previamente exportados |

### Nota sobre audio/video

El soporte de audio y video requiere la instalacion adicional:
```bash
pip install docling[asr]
```
y tener `ffmpeg` disponible en PATH para formatos de video.
Actualmente el servicio **no incluye** `docling[asr]` por defecto.
Si se requiere, agregar al `requirements.txt`.

---

## 3. Requisitos de infraestructura

### Hardware minimo

| Recurso         | Minimo             | Recomendado (produccion)        |
|-----------------|--------------------|---------------------------------|
| CPU             | 8 cores            | 16+ cores                       |
| RAM             | 32 GB              | 64 GB                           |
| GPU             | 1x NVIDIA (16+ GB) | 3x RTX A6000 (48 GB c/u)       |
| Disco           | 50 GB SSD          | 100 GB NVMe                     |

### Software

| Componente          | Version requerida                        |
|---------------------|------------------------------------------|
| OS                  | Ubuntu 22.04+ / Windows Server 2022+     |
| Python              | 3.10, 3.11, 3.12, 3.13 o 3.14           |
| NVIDIA Driver       | >= 535.x (CUDA 12.x compatible)          |
| CUDA Toolkit        | 12.6 (para RTX A6000 Ampere)             |
| cuDNN               | 9.x (incluido en torch wheel)            |
| PostgreSQL          | 14+ (acceso de red al cluster existente)  |

### Verificacion previa del driver NVIDIA

```bash
# Debe mostrar las GPUs y version del driver
nvidia-smi

# Salida esperada (ejemplo con 3x A6000):
# +-------------------------+
# | NVIDIA-SMI 535.xxx      |
# | Driver Version: 535.xxx |
# | CUDA Version: 12.6      |
# +-------------------------+
# | GPU 0: NVIDIA RTX A6000 |
# | GPU 1: NVIDIA RTX A6000 |
# | GPU 2: NVIDIA RTX A6000 |
# +-------------------------+
```

Si `nvidia-smi` no funciona, instalar el driver antes de continuar.

---

## 4. Instalacion paso a paso

### 4.1 Clonar el repositorio

```bash
git clone <URL_REPOSITORIO> brain-vt-ocr-chunking
cd brain-vt-ocr-chunking
```

### 4.2 Crear entorno virtual

```bash
python -m venv venv

# Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4.3 Instalar PyTorch con CUDA

**IMPORTANTE**: Este paso se hace ANTES de instalar requirements.txt.
La version de CUDA depende de la GPU:

```bash
# ============================================================
# PRODUCCION - RTX A6000 Ampere (sm_86) - usar cu126
# ============================================================
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ============================================================
# DESARROLLO - RTX 5090 Blackwell (sm_120) - usar cu130
# ============================================================
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 4.4 Verificar PyTorch + CUDA

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'Architectures: {torch.cuda.get_arch_list()}')
# Prueba rapida
t = torch.randn(256, 256, device='cuda')
print(f'Tensor en GPU: OK ({t.device})')
"
```

Salida esperada para RTX A6000:
```
PyTorch: 2.10.0+cu126
CUDA disponible: True
GPU: NVIDIA RTX A6000
Compute capability: (8, 6)
Architectures: ['sm_50', 'sm_60', ..., 'sm_86', 'sm_90']
Tensor en GPU: OK (cuda:0)
```

**Si CUDA no esta disponible**: el driver NVIDIA no esta instalado correctamente.
**Si sm_86 no aparece en la lista**: se instalo el torch incorrecto (verificar --index-url).

### 4.5 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4.6 Pre-descarga de modelos (offline/primera vez)

El servicio descarga modelos de HuggingFace la primera vez que se ejecuta.
Para evitar demoras en la primera peticion o si el cluster no tiene acceso
a internet, pre-descargar:

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
# Modelo de embeddings (~2.2 GB)
model_name = 'intfloat/multilingual-e5-large-instruct'
print(f'Descargando {model_name}...')
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
print('Modelo de embeddings descargado.')

# Modelos de docling (layout, table structure) se descargan al primer uso
from docling.document_converter import DocumentConverter
print('Inicializando docling (descarga modelos de layout)...')
converter = DocumentConverter()
print('Modelos de docling descargados.')
print('Pre-descarga completada.')
"
```

Los modelos se guardan en `~/.cache/huggingface/hub/`. Si se necesita
una ruta diferente, configurar la variable `HF_HOME` antes de ejecutar:

```bash
export HF_HOME=/ruta/modelos/huggingface
```

---

## 5. Variables de entorno

### Obligatorias

| Variable              | Descripcion                                     | Ejemplo                        |
|-----------------------|-------------------------------------------------|--------------------------------|
| `PG_HOST`             | Host de PostgreSQL                               | `10.0.1.50`                    |
| `PG_PORT`             | Puerto de PostgreSQL                             | `5432`                         |
| `PG_DATABASE`         | Nombre de la base de datos                       | `brain_vt`                     |
| `PG_USER`             | Usuario de PostgreSQL                            | `ocr_service`                  |
| `PG_PASSWORD`         | Password de PostgreSQL                           | `***`                          |
| `AUTH_JWKS_URL`       | URL del endpoint JWKS para validacion JWT        | `https://auth.example.com/.well-known/jwks.json` |
| `AUTH_AUDIENCE`       | Audience esperada en el token JWT                | `brain-vt-api`                 |

### Opcionales (con defaults)

| Variable                | Default   | Descripcion                                              |
|-------------------------|-----------|----------------------------------------------------------|
| `OCR_PORT`              | `80`      | Puerto del servicio                                      |
| `OCR_HOST`              | `0.0.0.0` | Interfaz de escucha                                     |
| `OCR_WORKERS`           | `1`       | Workers de uvicorn (mantener 1 para GPU)                 |
| `OCR_LOG_LEVEL`         | `INFO`    | Nivel de log (DEBUG, INFO, WARNING, ERROR)               |
| `OCR_PAGE_CHUNK_SIZE`   | `50`      | Paginas por bloque para PDFs grandes (0=desactivar)      |
| `OCR_DOCUMENT_TIMEOUT`  | `300`     | Timeout en segundos por bloque de paginas                |
| `OCR_LOG_DIR`           | `<tempdir>/ocr-chunking-logs` | Directorio de logs estructurados       |
| `OCR_LOG_RETENTION_DAYS`| `30`      | Dias de retencion de logs                                |
| `DEFAULT_EMBEDDING_MODEL`| `intfloat/multilingual-e5-large-instruct` | Modelo de embeddings   |
| `HF_HOME`               | `~/.cache/huggingface` | Directorio de cache de modelos HuggingFace |
| `CUDA_VISIBLE_DEVICES`  | (todas)   | GPUs visibles (ej: `0,1,2` o `0`)                       |

### Ejemplo de archivo .env

```bash
# === Produccion - Cluster Airflows ===
PG_HOST=10.0.1.50
PG_PORT=5432
PG_DATABASE=brain_vt
PG_USER=ocr_service
PG_PASSWORD=SecurePassword123
AUTH_JWKS_URL=https://auth.anh.gov.co/.well-known/jwks.json
AUTH_AUDIENCE=brain-vt-api
OCR_PORT=8080
OCR_HOST=0.0.0.0
OCR_LOG_LEVEL=INFO
OCR_PAGE_CHUNK_SIZE=100
OCR_DOCUMENT_TIMEOUT=600
CUDA_VISIBLE_DEVICES=0,1,2
HF_HOME=/opt/models/huggingface
```

---

## 6. Arranque del servicio

### 6.1 Ejecucion directa

```bash
# Linux
source venv/bin/activate
python ocr_chunking.py

# Windows
venv\Scripts\activate
python ocr_chunking.py
```

El servicio arranca en `http://{OCR_HOST}:{OCR_PORT}`.

### 6.2 Ejecucion con uvicorn (recomendado para produccion)

```bash
source venv/bin/activate
uvicorn ocr_chunking:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1 \
  --timeout-keep-alive 300 \
  --log-level info
```

**IMPORTANTE**: Usar `--workers 1`. Multiples workers cargan el modelo
en GPU multiples veces, agotando la VRAM. Para escalar, usar multiples
instancias del servicio con `CUDA_VISIBLE_DEVICES` diferente por instancia.

### 6.3 Ejecucion como servicio systemd (Linux)

Crear `/etc/systemd/system/ocr-chunking.service`:

```ini
[Unit]
Description=Brain VT OCR Chunking Service
After=network.target

[Service]
Type=simple
User=ocr-service
Group=ocr-service
WorkingDirectory=/opt/brain-vt-ocr-chunking
EnvironmentFile=/opt/brain-vt-ocr-chunking/.env
ExecStart=/opt/brain-vt-ocr-chunking/venv/bin/uvicorn \
  ocr_chunking:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 1 \
  --timeout-keep-alive 300
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-chunking
sudo systemctl start ocr-chunking
sudo systemctl status ocr-chunking
```

### 6.4 Escalamiento multi-GPU

Para aprovechar las 3 GPUs, ejecutar 3 instancias del servicio,
cada una asignada a una GPU diferente, detras de un balanceador:

```bash
# Instancia GPU 0
CUDA_VISIBLE_DEVICES=0 OCR_PORT=8080 uvicorn ocr_chunking:app --host 0.0.0.0 --port 8080 --workers 1 &

# Instancia GPU 1
CUDA_VISIBLE_DEVICES=1 OCR_PORT=8081 uvicorn ocr_chunking:app --host 0.0.0.0 --port 8081 --workers 1 &

# Instancia GPU 2
CUDA_VISIBLE_DEVICES=2 OCR_PORT=8082 uvicorn ocr_chunking:app --host 0.0.0.0 --port 8082 --workers 1 &
```

Luego configurar nginx o HAProxy para balancear entre los 3 puertos.

---

## 7. Validacion post-despliegue

Despues de arrancar el servicio, ejecutar estos endpoints en orden para
validar que todo funciona correctamente.

### 7.1 Verificar que el servicio responde

```bash
curl http://localhost:8080/docs
# Debe retornar la interfaz Swagger (HTTP 200)
```

### 7.2 Validar GPU

```bash
curl -H "Authorization: Bearer <TOKEN_JWT>" \
  http://localhost:8080/validate-gpu
```

Respuesta esperada (RTX A6000):
```json
{
  "status": "ok",
  "gpu": {
    "cuda_available": true,
    "device_count": 1,
    "current_device": 0,
    "device_name": "NVIDIA RTX A6000",
    "compute_capability": "8.6",
    "cuda_arch_list": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
    "vram_total_gb": 48.0,
    "vram_used_gb": 0.5,
    "vram_free_gb": 47.5,
    "torch_version": "2.10.0+cu126",
    "cuda_version": "12.6"
  }
}
```

**Verificar**:
- `cuda_available: true`
- `device_name` muestra la GPU correcta
- `compute_capability` es `8.6` para A6000
- `sm_86` esta en `cuda_arch_list`

### 7.3 Validar librerias

```bash
curl -H "Authorization: Bearer <TOKEN_JWT>" \
  http://localhost:8080/validate-libraries
```

Verifica versiones de todas las dependencias criticas.

### 7.4 Validar entorno completo

```bash
curl -H "Authorization: Bearer <TOKEN_JWT>" \
  http://localhost:8080/validate-environment
```

Verifica GPU + librerias + conectividad PostgreSQL + auth.

### 7.5 Test de stress GPU

```bash
curl -H "Authorization: Bearer <TOKEN_JWT>" \
  http://localhost:8080/validate-cuda-stress
```

Ejecuta operaciones matriciales en GPU para verificar estabilidad.

### 7.6 Test de pipeline completo

```bash
curl -X POST \
  -H "Authorization: Bearer <TOKEN_JWT>" \
  -F "file=@documento_prueba.pdf" \
  "http://localhost:8080/validate-pipeline/upload?chunking_strategy=semantic&max_chunks=5"
```

Ejecuta el pipeline completo (OCR -> limpieza -> chunking -> embeddings)
**sin persistir en BD**. Retorna un reporte detallado con tiempos por fase.

---

## 8. Endpoints disponibles

### Endpoints de negocio

| Metodo | Ruta                        | Descripcion                              |
|--------|----------------------------|------------------------------------------|
| POST   | `/PipelineOCR/process`     | Pipeline principal: procesa documento y persiste en BD |

### Endpoints de validacion

| Metodo | Ruta                          | Descripcion                           |
|--------|-------------------------------|---------------------------------------|
| GET    | `/validate-gpu`               | Estado de GPU, VRAM, CUDA             |
| GET    | `/validate-libraries`         | Versiones de dependencias             |
| GET    | `/validate-environment`       | GPU + libs + DB + auth                |
| GET    | `/validate-cuda-stress`       | Test de stress GPU                    |
| POST   | `/validate-pipeline/upload`   | Test e2e con archivo de prueba        |

### Endpoints de logs

| Metodo | Ruta                          | Descripcion                           |
|--------|-------------------------------|---------------------------------------|
| GET    | `/logs/summary`               | Resumen: archivos, errores, disco     |
| GET    | `/logs/list`                  | Lista logs con filtros                |
| GET    | `/logs/detail/{filename}`     | Contenido de un log especifico        |

### Autenticacion

Todos los endpoints (excepto `/docs` y `/openapi.json`) requieren un
token JWT en el header `Authorization: Bearer <token>`.

---

## 9. Arquitectura GPU y compatibilidad

### Tabla de compatibilidad GPU / CUDA / PyTorch

| GPU             | Arquitectura | Compute Cap. | CUDA requerido | Comando pip torch                                             |
|-----------------|-------------|--------------|----------------|---------------------------------------------------------------|
| RTX A6000       | Ampere      | sm_86        | cu124 / cu126  | `pip install torch torchvision --index-url .../whl/cu126`     |
| RTX 4090        | Ada Lovelace| sm_89        | cu124 / cu126  | `pip install torch torchvision --index-url .../whl/cu126`     |
| RTX 5090        | Blackwell   | sm_120       | cu128 / cu130  | `pip install torch torchvision --index-url .../whl/cu130`     |
| A100            | Ampere      | sm_80        | cu118 / cu126  | `pip install torch torchvision --index-url .../whl/cu126`     |
| H100            | Hopper      | sm_90        | cu124 / cu126  | `pip install torch torchvision --index-url .../whl/cu126`     |

### Como funciona la aceleracion GPU en el servicio

```
PDF (500 pags)
  |
  v
[PyMuPDF: Split en bloques de 50 pags]  <-- CPU
  |
  v (bloque 1)         v (bloque 2)         v (bloque N)
[docling-parse: C++]   [docling-parse: C++]  ...           <-- CPU + RAM
  |                     |
  v                     v
[Layout model: batch=64]                                    <-- GPU (CUDA)
  |                     |
  v                     v
[RapidOCR: torch backend, GPU]                              <-- GPU (CUDA)
  |                     |
  v                     v
[gc.collect() + cuda.empty_cache()]                         <-- Liberar memoria
  |
  v
[Concatenar textos]
  |
  v
[Chunking semantico]                                        <-- CPU
  |
  v
[Embeddings: multilingual-e5-large, float16]                <-- GPU (CUDA)
  |
  v
[PostgreSQL: INSERT fragmentos + vectores]                  <-- Red
```

### Fases CPU vs GPU

| Fase              | Recurso | Notas                                        |
|-------------------|---------|----------------------------------------------|
| docling-parse     | CPU+RAM | Codigo C++ nativo, no se puede mover a GPU   |
| Layout detection  | GPU     | Modelo Heron, batch_size=64                  |
| OCR (RapidOCR)    | GPU     | Backend torch, modelos .pth en VRAM          |
| Table structure   | GPU     | TableFormer, batch_size=4                    |
| Embeddings        | GPU     | float16 para reducir VRAM a la mitad         |
| Chunking/limpieza | CPU     | Operaciones de texto, instantaneo            |

---

## 10. Optimizacion de rendimiento

### Benchmark de referencia (112 paginas, PDF con OCR)

| Configuracion                     | Tiempo  | pag/seg |
|-----------------------------------|---------|---------|
| CPU only (onnxruntime, sin chunk) | 548 seg | 0.2     |
| GPU torch, sin chunking           | 140 seg | 0.8     |
| GPU torch + chunks de 50 pags     | 100 seg | 1.1     |

### Parametros de tuning

```bash
# Paginas por bloque (mayor = menos overhead, mas RAM)
# Recomendado: 50 (laptop 64GB), 100 (servidor 128GB+)
OCR_PAGE_CHUNK_SIZE=100

# Timeout por bloque (incrementar para PDFs muy complejos)
OCR_DOCUMENT_TIMEOUT=600

# Usar todas las GPUs (una instancia por GPU)
CUDA_VISIBLE_DEVICES=0  # Instancia 1
CUDA_VISIBLE_DEVICES=1  # Instancia 2
CUDA_VISIBLE_DEVICES=2  # Instancia 3
```

### RTX A6000 (48 GB VRAM) - Configuracion recomendada

Con 48 GB de VRAM (vs 24 GB del RTX 5090 laptop), se puede subir:
- `OCR_PAGE_CHUNK_SIZE=100` (mas paginas por bloque)
- Los batch sizes internos (layout=64, ocr=64) ya estan configurados
- `OCR_DOCUMENT_TIMEOUT=600` para documentos muy grandes

### Estimaciones de tiempo (con GPU + chunking)

| Paginas | Tiempo estimado (1 GPU) | Tiempo estimado (3 GPUs paralelo) |
|---------|-------------------------|-----------------------------------|
| 50      | ~40 seg                 | ~40 seg (1 documento)             |
| 100     | ~80 seg                 | ~80 seg (1 documento)             |
| 500     | ~420 seg (~7 min)       | ~7 min (1 doc) / ~2.5 min (3 docs)|
| 1000    | ~840 seg (~14 min)      | ~14 min (1 doc) / ~5 min (3 docs) |

Nota: Las 3 GPUs benefician el procesamiento **paralelo de multiples
documentos**, no la velocidad de un solo documento (que es secuencial).

---

## 11. Logs y observabilidad

### Logs estructurados en disco

El servicio escribe logs JSONL en:
- Linux: `/tmp/ocr-chunking-logs/` (o `OCR_LOG_DIR`)
- Windows: `%TEMP%\ocr-chunking-logs\`

Estructura:
```
{OCR_LOG_DIR}/
  2026-03-21/
    pipeline_2026-03-21T10-45-50_oid-12345.jsonl
    validation_2026-03-21T10-45-50.jsonl
    errors_2026-03-21.jsonl
  2026-03-20/
    ...
```

### Consultar logs via API

```bash
# Resumen general
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8080/logs/summary

# Listar logs de un dia
curl -H "Authorization: Bearer <TOKEN>" "http://localhost:8080/logs/list?date=2026-03-21&limit=50"

# Ver un log especifico
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8080/logs/detail/pipeline_2026-03-21T10-45-50_oid-12345.jsonl
```

### Stdout / journald

El servicio tambien escribe a stdout en formato estandar:
```
2026-03-21 10:45:50,197 | INFO | ocr_chunking | Large PDF detected (112 pages). Splitting into chunks of 50 pages.
2026-03-21 10:45:50,211 | INFO | ocr_chunking | Processing chunk 1/3 (pages 1-50)...
```

Con systemd, estos logs van a journald y se consultan con:
```bash
journalctl -u ocr-chunking -f
```

---

## 12. Troubleshooting

### Error: `CUDA error: no kernel image is available for execution on the device`

**Causa**: PyTorch instalado con CUDA incompatible con la GPU.

**Solucion**: Verificar compute capability y reinstalar torch:
```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Si sale (8, 6) -> RTX A6000 -> usar cu126
# Si sale (12, 0) -> RTX 5090 -> usar cu130
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### Error: `std::bad_alloc` en docling-parse

**Causa**: PDF con paginas muy pesadas que agotan la RAM.

**Solucion**: Reducir `OCR_PAGE_CHUNK_SIZE`:
```bash
export OCR_PAGE_CHUNK_SIZE=30
```
El servicio divide el PDF en bloques mas pequenos con GC entre cada uno.

### Error: `torch.cuda.OutOfMemoryError`

**Causa**: VRAM insuficiente para los modelos + datos.

**Solucion**:
1. Reducir batch sizes (editar las constantes en `load_docling_converter`)
2. Asegurar que solo 1 worker de uvicorn esta corriendo (`--workers 1`)
3. Verificar que no hay otros procesos usando la GPU: `nvidia-smi`

### Error: `ConnectionRefusedError` a PostgreSQL

**Causa**: Variables PG_* mal configuradas o firewall.

**Solucion**: Verificar conectividad:
```bash
python -c "
import psycopg2
conn = psycopg2.connect(host='$PG_HOST', port=$PG_PORT, dbname='$PG_DATABASE', user='$PG_USER', password='$PG_PASSWORD')
print('Conexion OK')
conn.close()
"
```

### El servicio es lento (GPU al 0%)

**Causa probable**: PyTorch no detecta CUDA.

**Diagnostico**:
```bash
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8080/validate-gpu
```
Verificar que `cuda_available: true` y que `device_name` muestra la GPU.

### Warning: `torch_dtype is deprecated`

**Inocuo**: Ya corregido en el codigo. Si aparece, significa que una
dependencia interna de transformers usa la sintaxis antigua. No afecta
el funcionamiento.

### Warning: `Token indices sequence length is longer than the specified maximum sequence length`

**Inocuo**: El tokenizer avisa que el texto completo excede 512 tokens,
pero el chunking semantico ya lo divide en fragmentos mas pequenos antes
de generar embeddings. No requiere accion.

---

## 13. Despliegue con Docker

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch with CUDA 12.6 (RTX A6000 Ampere)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct'); \
AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')"

# Copy application
COPY ocr_chunking.py .

EXPOSE 8080

CMD ["uvicorn", "ocr_chunking:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "300"]
```

### docker-compose.yml (3 GPUs)

```yaml
version: "3.9"

services:
  ocr-gpu0:
    build: .
    ports:
      - "8080:8080"
    env_file: .env
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OCR_PORT=8080
      - OCR_PAGE_CHUNK_SIZE=100
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    restart: unless-stopped

  ocr-gpu1:
    build: .
    ports:
      - "8081:8080"
    env_file: .env
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OCR_PORT=8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    restart: unless-stopped

  ocr-gpu2:
    build: .
    ports:
      - "8082:8080"
    env_file: .env
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OCR_PORT=8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
    restart: unless-stopped
```

```bash
docker compose up -d --build
```

---

## Checklist de despliegue

- [ ] `nvidia-smi` muestra las GPUs correctamente
- [ ] Python 3.10+ instalado
- [ ] `torch.cuda.is_available()` retorna `True`
- [ ] `sm_86` aparece en `torch.cuda.get_arch_list()` (para A6000)
- [ ] Variables de entorno configuradas (.env)
- [ ] Conectividad a PostgreSQL verificada
- [ ] Modelos pre-descargados (o acceso a internet habilitado)
- [ ] Servicio arranca sin errores
- [ ] `/validate-gpu` retorna `status: ok`
- [ ] `/validate-libraries` retorna todas las versiones correctas
- [ ] `/validate-pipeline/upload` completa exitosamente con PDF de prueba
- [ ] Logs se escriben en `OCR_LOG_DIR`
- [ ] Firewall permite trafico al puerto del servicio
