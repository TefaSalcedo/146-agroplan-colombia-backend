# AgroPlan Colombia Backend

Backend FastAPI con Docker y PostgreSQL 18 que sirve datos a las pantallas del frontend AgroPlan Colombia y expone endpoints de predicción con estructura lista para conectar modelos LightGBM, CatBoost y XGBoost.

## Stack Tecnológico

- **Python 3.14-slim**
- **FastAPI >= 0.115.0**
- **PostgreSQL 18**
- **SQLAlchemy 2.0 + Alembic**
- **Open-Meteo API** (clima, sin API key)
- **Docker Compose**

## Estructura del Proyecto

```
backend-agroplan-colombia/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── database.py          # SQLAlchemy engines
│   ├── dependencies.py      # FastAPI dependencies
│   ├── models.py            # SQLAlchemy ORM models
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   └── schemas/             # Pydantic models
├── alembic/                 # Database migrations
├── data/                    # Static data (DIVIPOLA)
├── scripts/                 # Utility scripts
├── tests/                   # Pytest test suite
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Setup desde Cero con Docker

### 1. Prerrequisitos

- Docker 24+
- Docker Compose v2+
- `curl` o navegador para probar endpoints

### 2. Clonar y entrar al proyecto

```bash
git clone <repo-url>
cd 146-AgroPlan-Colombia-Backend
```

### 3. Copiar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` con los valores que necesites. Como mínimo:

```bash
# Base de datos local (Docker)
DATABASE_URL=postgresql://agroplan:agroplan@db:5432/agroplan
MIGRATION_DATABASE_URL=postgresql://agroplan:agroplan@db:5432/agroplan

# API
ADMIN_API_KEY=tu-admin-key-segura
API_V1_PREFIX=/api/v1

# Desactivar sync automático en primera carga
ENABLE_CLIMATE_SYNC=false
```

### 4. Construir imágenes y levantar servicios

```bash
docker compose up -d --build
```

Esto levanta:
- `db`: PostgreSQL 18 en `localhost:5432`
- `api`: FastAPI en `http://localhost:8000`

### 5. Ejecutar migraciones de Alembic

```bash
docker compose exec api alembic upgrade head
```

### 6. Poblar datos iniciales

```bash
docker compose exec api python scripts/seed_db.py --force --strict
```

Verifica la cobertura:

```bash
# Debe reportar 1122 municipios y 33 departamentos
```

### 7. Verificar health

```bash
curl http://localhost:8000/api/v1/health
```

### 8. Reiniciar el contenedor para sincronización (opcional)

Si quieres que el job de clima se ejecute automáticamente, activa `ENABLE_CLIMATE_SYNC=true` en `.env`:

```bash
docker compose down
docker compose up -d
```

Para una carga inicial completa de clima:

```bash
docker compose exec api python scripts/seed_db.py --force --strict --sync-climate --limit 50
```

> Nota: una carga completa de ~1.100 municipios consume aproximadamente 1.100 requests y tarda varias horas. Recomendamos `--limit 50` para pruebas.

## Desarrollo

### Levantar solo la base de datos

```bash
docker compose up -d db
```

### Migrar base de datos local

```bash
python -m alembic upgrade head
```

### Poblar datos

```bash
python scripts/seed_db.py --force --strict
```

### Correr tests

```bash
python -m pytest tests/ -v
```

### Logs

```bash
docker compose logs -f api
docker compose logs -f db
```

Puedes aumentar el detalle de los logs con la variable `LOG_LEVEL` en `.env`:

```bash
LOG_LEVEL=DEBUG
```

- `DEBUG`: muestra paso a paso cada endpoint, consulta a base de datos, llamada a modelos ML, carga de perfiles/parquets, llamadas a LLM y sincronización con Open-Meteo.
- `INFO` (default): muestra entradas/salidas de endpoints, eventos de modelos y resúmenes de jobs.
- `WARNING`/`ERROR`: solo errores y advertencias.

Reinicia el contenedor para aplicar el cambio:

```bash
docker compose restart api
```

## Variables de Entorno

| Variable | Descripción | Ejemplo |
|---|---|---|
| `DATABASE_URL` | URL de conexión pooled para la app | `postgresql://agroplan:agroplan@db:5432/agroplan` |
| `MIGRATION_DATABASE_URL` | URL directa para Alembic (sin pooling) | `postgresql://agroplan:agroplan@db:5432/agroplan` |
| `OPEN_METEO_BASE_URL` | Base URL del API de forecast | `https://api.open-meteo.com` |
| `OPEN_METEO_ARCHIVE_URL` | Base URL del API de archivo | `https://archive-api.open-meteo.com` |
| `ML_MODELS_PATH` | Ruta local de artefactos ML | `./models` |
| `HF_TOKEN` | Token de Hugging Face | `hf_...` |
| `HF_MODEL_REPO_ZONING` | Repo de modelos de zonificación | `agroplan/zoning-models` |
| `HF_MODEL_REPO_YIELD` | Repo de modelos de rendimiento | `agroplan/yield-models` |
| `HF_MODEL_REVISION` | Revisión fija de HF | `main` |
| `LLM_PROVIDER` | Punto de inicio del round-robin (`openrouter` o `groq`) | `openrouter` |
| `OPENROUTER_API_KEY` | API key de OpenRouter | `sk-...` |
| `OPENROUTER_MODELS` | Modelos separados por coma (round-robin) | `google/gemini-2.0-flash:free,meta-llama/llama-3.3-70b-instruct:free` |
| `GROQ_API_KEY` | API key de Groq | `gsk_...` |
| `GROQ_MODELS` | Modelos separados por coma (round-robin) | `llama-3.3-70b-versatile,llama-3.1-8b-instant,mistral-saba-24b` |
| `LLM_TIMEOUT_SECONDS` | Timeout por llamada LLM | `30` |
| `ADMIN_API_KEY` | API key para endpoints admin | `tu-admin-key-segura` |
| `CACHE_TTL_STRATEGY` | Estrategia de TTL de caché | `end_of_month_bogota` |
| `ENABLE_MOCK_PREDICTOR` | Activar MockPredictor como fallback final | `false` |
|| `LOG_LEVEL` | Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `API_V1_PREFIX` | Prefijo de la API | `/api/v1` |
| `CORS_ORIGINS` | Orígenes permitidos | `http://localhost:3000,http://localhost:3001` |
| `ENABLE_CLIMATE_SYNC` | Habilitar sync programado | `true` |
| `CLIMATE_SYNC_HOUR` | Hora del sync diario | `3` |
| `CLIMATE_SYNC_MINUTE` | Minuto del sync diario | `0` |
| `CLIMATE_SYNC_BATCH_SIZE` | Tamaño de lote | `100` |
| `CLIMATE_SYNC_DELAY_SECONDS` | Delay entre requests | `2.0` |
| `CLIMATE_SYNC_DAYS_AHEAD` | Días de forecast | `16` |
| `CLIMATE_SYNC_CLEANUP_DAYS` | Días a conservar | `180` |

## Endpoints

### System

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Metadata de la API |
| `GET` | `/api/v1/health` | Health check básico |
| `GET` | `/api/v1/readiness` | Readiness detallado por componente |

### Municipalities

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/municipalities` | Lista de municipios |
| `GET` | `/api/v1/municipalities?department_id=05` | Filtrar por DANE de departamento |
| `GET` | `/api/v1/municipalities?department=Antioquia` | Filtrar por nombre de departamento |
| `GET` | `/api/v1/municipalities/departments` | Lista de departamentos con DANE y conteo |
| `GET` | `/api/v1/municipalities/nearby?lat=6.24&lng=-75.58&max_distance_km=20` | Municipio más cercano a coordenadas |
| `GET` | `/api/v1/municipalities/{municipality_id}` | Detalle de un municipio (DANE 5 dígitos) |

### Weather & Forecast

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/weather/{municipality_id}` | Clima actual (BD con fallback a Open-Meteo) |
| `GET` | `/api/v1/forecast/daily/{municipality_id}?days=7` | Pronóstico diario (BD con fallback a Open-Meteo) |
| `GET` | `/api/v1/alerts/{municipality_id}` | Alertas climáticas corto plazo (BD con fallback a Open-Meteo) |

### Crops

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/crops` | Catálogo completo de cultivos (7 soportados) |
| `GET` | `/api/v1/crops/lite` | Lista ligera de cultivos |
| `GET` | `/api/v1/crops/{crop_id}` | Ficha de un cultivo |

### Zoning

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/zoning/recommendations/{municipality_id}` | Ranking de cultivos para un municipio (LightGBM + recomendaciones por clima/suelo) |
| `GET` | `/api/v1/zoning/map/{crop_id}` | Mapa de zonificación para todos los municipios |

### Calendars

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/calendars/predict` | Calendario legado un mes |
| `POST` | `/api/v1/calendars/predict-batch` | Calendario multi-cultivo 12 meses |


### Admin (requieren `X-Admin-API-Key`)

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/admin/climate-sync/status` | Estado del sync de clima |
| `GET` | `/api/v1/admin/models/status` | Estado de modelos ML cargados |
| `GET` | `/api/v1/admin/cache/stats` | Estadísticas de caché |
| `POST` | `/api/v1/admin/cache/invalidate` | Invalidar caché |
| `GET` | `/api/v1/admin/audit/recent` | Auditoría reciente de predicciones |

## Ejemplos de Uso

### Health

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "version": "2.0.0",
  "models_loaded": false
}
```

### Readiness

```bash
curl http://localhost:8000/api/v1/readiness
```

```json
{
  "status": "ok",
  "version": "2.0.0",
  "components": [
    { "name": "database", "ready": true, "detail": null },
    { "name": "zoning_model", "ready": false, "detail": "Model not loaded" },
    { "name": "yield_models", "ready": false, "detail": "Yield models not loaded" },
    { "name": "reference_profiles", "ready": false, "detail": "Profiles not loaded" },
    { "name": "golden_vectors", "ready": false, "detail": "Not validated" }
  ]
}
```

### Listar departamentos

```bash
curl http://localhost:8000/api/v1/municipalities/departments
```

```json
{
  "departments": ["ANTIOQUIA", "BOYACÁ", ...],
  "departments_detailed": [
    {
      "dane_code": "05",
      "name": "ANTIOQUIA",
      "municipality_count": 125
    }
  ],
  "count": 33
}
```

### Municipio por DANE

```bash
curl http://localhost:8000/api/v1/municipalities/05001
```

```json
{
  "id": "05001",
  "name": "MEDELLÍN",
  "department": "ANTIOQUIA",
  "department_id": "05",
  "lat": 6.246631,
  "lng": -75.581775,
  "altitude": 0,
  "avg_temperature": 0.0,
  "precipitation": 0.0,
  "dane_code": "05001"
}
```

### Listar cultivos

```bash
curl http://localhost:8000/api/v1/crops
```

Retorna los 7 cultivos soportados: `aguacate`, `algodon`, `cana_panelera`, `cebolla`, `fresa`, `pina`, `soya`.

### Zonificación y recomendación por municipio

```bash
curl http://localhost:8000/api/v1/zoning/recommendations/05001
```

```json
{
  "municipality_id": "05001",
  "municipality_name": "MEDELLÍN",
  "results": [
    {
      "crop_id": "aguacate",
      "crop_name": "Aguacate",
      "suitability": "medium",
      "confidence": 0.4414,
      "model_version": "zoning-lightgbm-v1",
      "method": "primary_model",
      ...
    }
  ],
  "climate_based_recommendations": [
    { "crop_id": "aguacate", "crop_name": "Aguacate", "score": 0.3333, "source": "climate_analog_knn" },
    { "crop_id": "cebolla", "crop_name": "Cebolla", "score": 0.2222, "source": "climate_analog_knn" }
  ],
  "model_version": "zoning-lightgbm-v1"
}
```

### Mapa de zonificación para un cultivo

```bash
curl http://localhost:8000/api/v1/zoning/map/aguacate
```

### Calendario batch

```bash
curl -X POST http://localhost:8000/api/v1/calendars/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "municipality_id": "05001",
    "crop_ids": ["aguacate", "pina"],
    "horizon_months": 12
  }'
```

### Admin: estado de modelos

```bash
curl -H "X-Admin-API-Key: tu-admin-key-segura" \
  http://localhost:8000/api/v1/admin/models/status
```

### Admin: estadísticas de caché

```bash
curl -H "X-Admin-API-Key: tu-admin-key-segura" \
  http://localhost:8000/api/v1/admin/cache/stats
```

### Admin: invalidar caché

```bash
curl -X POST http://localhost:8000/api/v1/admin/cache/invalidate \
  -H "X-Admin-API-Key: tu-admin-key-segura" \
  -H "Content-Type: application/json" \
  -d '{"prediction_type": "zoning"}'
```

## Integración de Modelos ML

### Opción A: Modelos locales en `models/`

```
models/
├── zoning/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_schema.json
│   └── manifest.json
├── yield/
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── preprocessor.pkl
│   └── feature_schema.json
└── zoning_reference.parquet
```

### Opción B: Modelos desde Hugging Face

Configura en `.env`:

```bash
HF_TOKEN=hf_...
HF_MODEL_REPO_ZONING=SRBOTOM/agroplan-zonificacion
HF_MODEL_REPO_YIELD=SRBOTOM/agroplan-rendimiento
HF_MODEL_REVISION=main
```

Al iniciar, `app/services/model_loader.py` descarga automáticamente todos los archivos de los repos configurados a `./models/zoning/` y `./models/yield/`. El token solo necesita permiso de lectura. Si un repo es privado, el token es obligatorio; si es público, puedes omitirlo.

El ensamble de rendimiento usa XGBoost con peso 0.65 y LightGBM con peso 0.35. Estos pesos pueden sobrescribirse creando `models/yield/weights.json`:

```json
{"xgboost": 0.65, "lightgbm": 0.35}
```

> **Nota:** los modelos publicados en HF son los estimadores finales. Para inferencia real también se requieren `preprocessor.pkl`, `feature_schema.json` y los perfiles Parquet (`municipality_profiles.parquet`, `yield_profiles.parquet`, `zoning_reference.parquet`). Mientras falten, el backend funciona con mocks y el campo `method` indica `mock`.

### Manifiesto de modelos

Ejemplo de `manifest.json`:

```json
{
  "model_type": "zoning",
  "model_family": "lightgbm",
  "hf_repo": "agroplan/zoning-models",
  "hf_revision": "abc1234",
  "artifact_filename": "model.pkl",
  "sha256": "abcdef123456...",
  "preprocessor_version": "1.0",
  "feature_schema": "feature_schema.json"
}
```

## Sincronización de Clima (Open-Meteo)

El backend incluye un job programado que consulta Open-Meteo por coordenadas para cada municipio y guarda los pronósticos en PostgreSQL.

### Variables almacenadas

- Temperatura mínima, máxima y promedio
- Precipitación
- Humedad relativa
- Índice UV máximo
- Velocidad del viento

### Ejecución

1. **Población inicial** (configurable):

```bash
docker compose exec api python scripts/seed_db.py --force --strict --sync-climate --limit 50
```

2. **Job programado** (dentro del contenedor Docker):

- Actualización diaria: refresca los próximos 7 días y limpia datos antiguos.
- Extensión semanal: agrega 7 días adicionales al horizonte.
- Configurable por variables de entorno en `.env`.

### Monitoreo

```bash
curl http://localhost:8000/api/v1/admin/climate-sync/status
```

### Estrategia de respeto al límite gratuito

- Una sola llamada a Open-Meteo cubre hasta 16 días de forecast por municipio.
- Los municipios se procesan en lotes con delays configurables.
- La carga inicial completa (~1.100 municipios) consume aproximadamente 1.100 requests.
- El mantenimiento diario consume ~1.100 requests adicionales, dentro del límite de 10.000/día.

## Despliegue en Oracle Cloud Free Tier

La imagen está preparada para correr en la capa gratuita de Oracle Cloud:

- Usa imágenes oficiales multi-arquitectura (`python:3.14-slim` y `postgres:18-alpine`), compatibles con Ampere A1 (ARM64).
- El modo por defecto `HF_DOWNLOAD_MODE=mvp` descarga solo los modelos estrictamente necesarios, evitando el archivo grande de Random Forest (~1.2 GB) que no se usa en el MVP.
- El volumen `models_data` persiste los modelos entre reinicios, por lo que no se vuelven a descargar después del primer arranque.

### Recursos recomendados

| Servicio | Shape recomendado | Notas |
|---|---|---|
| VM | `VM.Standard.A1.Flex` (hasta 4 OCPU / 24 GB RAM) | Suficiente para API + PostgreSQL + modelos MVP (~120 MB en memoria). |
| Alternativa | `VM.Standard.E2.1.Micro` (1/8 OCPU, 1 GB RAM) | Posible pero ajustada; considera desactivar el scheduler (`ENABLE_CLIMATE_SYNC=false`) y usar una base externa. |

### `.env` sugerido para Oracle

```bash
HF_DOWNLOAD_MODE=mvp
ENABLE_CLIMATE_SYNC=false
# Opcional: base de datos gestionada o en el mismo host
DATABASE_URL=postgresql://agroplan:agroplan@db:5432/agroplan
```

### Primer arranque en Oracle

```bash
# Clonar, copiar env y levantar
git clone <repo>
cd 146-AgroPlan-Colombia-Backend
cp .env.example .env
# editar .env con HF_TOKEN y demás variables
docker compose up -d --build

# Migrar y poblar
docker compose exec api alembic upgrade head
docker compose exec api python scripts/seed_db.py --force --strict
```

Tras la primera descarga, los modelos quedan en el volumen `models_data`. Los reinicios posteriores serán rápidos.

## Notas Importantes

- Los endpoints de predicción usan `mock` cuando no hay modelos cargados. El campo `method` indica `mock` o `primary_model`.
- La caché de predicciones expira al final del mes en `America/Bogota`.
- Cada predicción se audita en `prediction_runs`.
- Si el LLM no está disponible, el endpoint de calendario retorna `llm_status: "llm_unavailable"` y `explanation: null` con status 200.
- Los endpoints admin requieren `X-Admin-API-Key`.

## Base de Datos y Migraciones

### Crear una nueva migración

```bash
docker compose exec api alembic revision --autogenerate -m "descripcion"
```

### Aplicar migraciones

```bash
docker compose exec api alembic upgrade head
```

### Downgrade

```bash
docker compose exec api alembic downgrade -1
```

## Tests

```bash
# Tests completos
docker compose exec api python -m pytest tests/ -v

# Tests con coverage
docker compose exec api python -m pytest tests/ --cov=app --cov-report=term-missing
```

## Troubleshooting

### El contenedor `api` no arranca

```bash
docker compose logs api
```

### PostgreSQL no arranca con PG18

Si el error menciona `/var/lib/postgresql/data`, asegúrate de tener el `docker-compose.yml` actual con el volumen montado en `/var/lib/postgresql` (no `/var/lib/postgresql/data`).

### Modelos no se cargan

```bash
docker compose exec api python -c "from app.services.model_loader import get_model_loader; print(get_model_loader().get_status())"
```

### Limpieza total

```bash
docker compose down -v
docker compose up -d --build
```
