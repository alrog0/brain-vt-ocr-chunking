#!/usr/bin/env bash
# Arranca ocr_chunking.py con variables de entorno de BD.
# Uso: ./run_server.sh
#      o: source .env.local 2>/dev/null; ./run_server.sh
# Configura OCR_DB_* en el entorno o en .env.local (export VAR=valor).

set -e
cd "$(dirname "$0")"

# Cargar .env.local si existe (export VAR=valor por línea)
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi

: "${OCR_DB_HOST:=localhost}"
: "${OCR_DB_PORT:=5432}"
: "${OCR_DB_NAME:=niledb}"
: "${OCR_DB_USER:=postgres}"
export OCR_DB_HOST OCR_DB_PORT OCR_DB_NAME OCR_DB_USER OCR_DB_PASSWORD

exec python3 ocr_chunking.py --host 127.0.0.1 --port 8000 "$@"
