# Brain VT – OCR Chunking Embeddings

> Servicio de orquestación **OCR + chunking + embeddings** para el proyecto **Brain VT** (cliente ANH).  
> EKRADYON · GitHub Enterprise.

---

## Descripción

Servicio **FastAPI/OpenAPI** que orquesta:

- **OCR**: Docling o PyMuPDF (detección automática por confianza de texto).
- **Limpieza**: cabeceras repetidas, números aislados, opcional corrección de tokens invertidos.
- **Chunking**: semántico (Docling HybridChunker) o simple por tamaño/solapamiento.
- **Embeddings**: modelo `intfloat/multilingual-e5-large-instruct`, persistencia en `IaCore.Embeddings`.
- **Colas**: integración con `Operaciones.ColasProcesamiento` y `Operaciones.JobsProcesamiento` (tipo `BRAINVT_OCR_EMBEDDINGS_GPU`).

**Entrada principal:** OID de PostgreSQL Large Object (`pg_largeobject`). Los ítems se resuelven desde `Operaciones.ItemsIngestaSmb` por `loOid`.

---

## Ramas

- **main**: código estable; no hacer commit directo. Solo recibe merges desde `develop` o hotfixes.
- **develop**: rama de trabajo; integración y PR hacia `main` cuando corresponda.

---

## Requisitos

- Python 3.10+
- PostgreSQL con esquemas `Operaciones`, `IaCore`, `BrainVtCommons`.
- Opcional: GPU (CUDA) para embeddings.

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

**Servidor API (puerto 8000):**

```bash
python ocr_chunking.py --host 0.0.0.0 --port 8000
```

**Demo local sin base de datos (modo mock):**

```bash
python ocr_chunking.py --mock-local
```

**Variables de entorno (ejemplo):**

| Variable | Descripción | Default |
|----------|-------------|---------|
| `OCR_DB_HOST` | Host PostgreSQL | `localhost` |
| `OCR_DB_PORT` | Puerto | `5432` |
| `OCR_DB_NAME` | Base de datos | `niledb` |
| `OCR_DB_USER` | Usuario | `postgres` |
| `OCR_DB_PASSWORD` | Contraseña | — |
| `OCR_LOG_LEVEL` | Nivel de log | `INFO` |

---

## Endpoints OpenAPI

- `GET /health` — Estado del servicio y GPU.
- `GET /example-request` — Ejemplo de payload de solicitud.
- `POST /ocr-chunking/process` — Procesar un documento (entrada: `oid` y opciones).
- `POST /ocr-chunking/process-batch` — Procesar varios documentos (paralelo opcional).
- `GET /ocr-chunking/jobs/{job_id}` — Consultar estado de un job.

Documentación interactiva: `http://<host>:8000/docs` (Swagger UI).

---

## Estructura del repo

```
brain-vt-ocr-chunking/
├── ocr_chunking.py    # Servicio único (FastAPI + pipeline)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Proyecto y cliente

- **Proyecto:** Brain VT  
- **Cliente:** ANH  
- **Organización:** EKRADYON  
- **Uso:** interno; no distribuir sin autorización.
