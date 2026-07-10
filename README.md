# AgroPlan Colombia Backend

Backend FastAPI con Docker y PostgreSQL 16 que sirve datos a las pantallas del frontend AgroPlan Colombia y expone endpoints de predicción con estructura lista para conectar los modelos XGBoost.

## Stack Tecnológico

- **Python 3.12-slim**
- **FastAPI >= 0.115.0**
- **PostgreSQL 16**
- **Open-Meteo API** (clima, sin API key)
- **Docker Compose**

## Estructura del Proyecto

```
backend-agroplan-colombia/
 app/
    main.py              # FastAPI app
    config.py            # Settings
    database.py          # SQLAlchemy
    dependencies.py      # Dependencies
    routers/             # API endpoints
    services/            # Business logic
    schemas/             # Pydantic models
 data/                    # Static data
 scripts/                 # Utility scripts
 tests/                   # Tests (empty for now)
 docker-compose.yml
```

## Endpoints

### Health
- `GET /api/v1/health` - Health check

### Municipalities
- `GET /api/v1/municipalities` - Lista de municipios
- `GET /api/v1/municipalities?department={dept}` - Filtrar por departamento

### Weather
- `GET /api/v1/weather/{municipality_id}` - Clima actual desde Open-Meteo

### Crops
- `GET /api/v1/crops` - Catálogo de cultivos
- `GET /api/v1/crops/{id}` - Ficha de un cultivo

### Predictions (Mock)
- `POST /api/v1/zoning/predict` - Zonificación agroclimática
- `POST /api/v1/calendars/predict` - Calendario de siembra
- `POST /api/v1/recommendations` - Recomendaciones de cultivos

## Setup

### 1. Copiar variables de entorno
```bash
cp .env.example .env
```

### 2. Configurar ML_DATA_PATH
En `.env`, ajustar `ML_DATA_PATH` apuntando a la carpeta `3_data_preparation/data/processed` del repo `agroplan-colombia`.

### 3. Iniciar con Docker
```bash
docker-compose up -d
```

### 4. Poblar la base de datos
```bash
docker-compose exec api python scripts/seed_db.py --complete --strict
```

### 5. Verificar health
```bash
curl http://localhost:8000/api/v1/health
```

## Desarrollo

El servidor se reinicia automáticamente con `--reload` en modo desarrollo.

Logs:
```bash
docker-compose logs -f api
```

## Notas

- Los endpoints de predicción usan mock basado en reglas simples (altitud, temperatura, precipitación) y ahora leen datos reales de forecast de Open-Meteo.
- Cuando el equipo de ML entregue los modelos `.pkl`, se actualiza `app/services/model_loader.py` y `app/services/mock_predictor.py`.
- Open-Meteo no requiere API key (límite 10K requests/día).

## Sincronización de Clima (Open-Meteo)

El backend incluye un job programado que consulta Open-Meteo por coordenadas para cada municipio y guarda los pronósticos en PostgreSQL.

### Variables almacenadas
- Temperatura mínima, máxima y promedio
- Precipitación
- Humedad relativa
- Índice UV máximo
- Velocidad del viento

### Ejecución

1. **Población inicial** (3 meses):
```bash
docker-compose exec api python scripts/seed_db.py --force --strict --sync-climate
```

2. **Job programado** (dentro del contenedor Docker):
- Actualización diaria: refresca los próximos 7 días y limpia datos antiguos.
- Extensión semanal: agrega 7 días adicionales al horizonte.
- Configurable por variables de entorno en `.env`:
  - `ENABLE_CLIMATE_SYNC=true`
  - `CLIMATE_SYNC_HOUR=3`
  - `CLIMATE_SYNC_MINUTE=0`
  - `CLIMATE_SYNC_BATCH_SIZE=100`
  - `CLIMATE_SYNC_DELAY_SECONDS=2.0`
  - `CLIMATE_SYNC_DAYS_AHEAD=90`
  - `CLIMATE_SYNC_CLEANUP_DAYS=180`

### Monitoreo
```bash
curl http://localhost:8000/api/v1/admin/climate-sync/status
```

### Estrategia de respeto al límite gratuito
- Una sola llamada a Open-Meteo cubre hasta 90 días de forecast por municipio.
- Los municipios se procesan en lotes de 100 con delays de 2 segundos.
- La carga inicial completa (~1.100 municipios) consume aproximadamente 1.100 requests.
- El mantenimiento diario consume ~1.100 requests adicionales, dentro del límite de 10.000/día.

## Integración de Modelos ML

Cuando los modelos XGBoost estén entrenados:

### 1. Colocar los modelos
Colocar los archivos `.pkl` o `.joblib` en la carpeta `models/` del backend:
```
models/
 zoning_model.pkl      # Modelo de zonificación (XGBoost Classifier)
 calendar_model.pkl    # Modelo de calendarios (XGBoost Regressor)
```

### 2. Actualizar `app/services/model_loader.py`
Descomentar y ajustar las rutas para cargar los modelos:
```python
self.zoning_model = joblib.load("models/zoning_model.pkl")
self.calendar_model = joblib.load("models/calendar_model.pkl")
```

### 3. Actualizar `app/services/mock_predictor.py`
Reemplazar la lógica de mock por inferencia real:
```python
def predict_zoning(self, crop_id, municipality_id, features):
    model = model_loader.get_zoning_model()
    prediction = model.predict(features)
    return prediction
```

### 4. Actualizar health endpoint
En `app/main.py`, cambiar `models_loaded` a `True` cuando los modelos estén cargados.

## Contrato API Completo

### GET /api/v1/health
Response:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": false
}
```

### GET /api/v1/municipalities?department=Antioquia
Response:
```json
{
  "municipalities": [
    {
      "id": "rionegro",
      "name": "Rionegro",
      "department": "Antioquia",
      "lat": 6.155,
      "lng": -75.374,
      "altitude": 2125,
      "avg_temperature": 17,
      "precipitation": 1900,
      "dane_code": "05660"
    }
  ],
  "count": 1
}
```

### GET /api/v1/weather/{municipality_id}
Response:
```json
{
  "temperature": 19.0,
  "condition": "Parcialmente nublado",
  "humidity": 72,
  "precipitation": 45,
  "icon": "partly",
  "source": "open-meteo",
  "fetched_at": "2026-07-02T19:30:00Z"
}
```

### GET /api/v1/crops
Response:
```json
{
  "crops": [
    {
      "id": "cafe",
      "name": "Café",
      "scientific_name": "Coffea arabica",
      "image": "/crops/cafe.png",
      "success_rate": 92,
      "recommendation": "high",
      "short_reason": "Clima y altitud ideales en tu zona.",
      "reason": "...",
      "days_to_harvest": 270,
      "soil_type": "Franco, fértil y bien drenado",
      "ideal_temperature": "18  24 °C",
      "humidity": "70  80 %",
      "precipitation": "1.500  2.500 mm / año",
      "altitude": "1.200  2.000 msnm",
      "irrigation": "Moderado, mantener humedad constante",
      "substrates": ["Materia orgánica", "Compost", "Cascarilla de arroz"],
      "planting_months": [2, 3, 9, 10],
      "harvest_months": [4, 5, 10, 11],
      "stages": [...],
      "tips": [...]
    }
  ],
  "count": 8
}
```

### POST /api/v1/zoning/predict
Request:
```json
{
  "crop_id": "cafe",
  "municipality_id": "rionegro"
}
```
Response:
```json
{
  "crop_id": "cafe",
  "municipality_id": "rionegro",
  "suitability": "high",
  "confidence": 0.92,
  "model_version": "mock-v1",
  "factors": {
    "temperature_match": true,
    "precipitation_match": true,
    "soil_match": true,
    "altitude_match": true
  }
}
```

### POST /api/v1/calendars/predict
Request:
```json
{
  "crop_id": "cafe",
  "municipality_id": "rionegro",
  "month": 7,
  "year": 2026
}
```
Response:
```json
{
  "crop_id": "cafe",
  "municipality_id": "rionegro",
  "month": 7,
  "year": 2026,
  "days": [
    {"day": 1, "rating": "ideal"},
    {"day": 2, "rating": "acceptable"},
    ...
  ],
  "ideal_count": 12,
  "model_version": "mock-v1"
}
```

### POST /api/v1/recommendations
Request:
```json
{
  "municipality_id": "rionegro"
}
```
Response:
```json
{
  "top_crop": {
    "id": "cafe",
    "name": "Café",
    ...
    "suitability": "high"
  },
  "other_crops": [
    {
      "id": "maiz",
      "name": "Maíz",
      "image": "/crops/maiz.png",
      "recommendation": "high",
      "success_rate": 85
    }
  ],
  "next_planting_season": {
    "month": 9,
    "month_name": "Septiembre",
    "crops": ["cafe", "maiz"]
  }
}
```

