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
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── database.py          # SQLAlchemy
│   ├── dependencies.py      # Dependencies
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   └── schemas/             # Pydantic models
├── data/                    # Static data
├── scripts/                 # Utility scripts
├── tests/                   # Tests (empty for now)
└── docker-compose.yml
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
docker-compose exec api python scripts/seed_db.py
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

- Los endpoints de predicción usan mock basado en CSVs procesados del repo ML.
- Cuando el equipo de ML entregue los modelos `.pkl`, solo se actualiza `app/services/model_loader.py`.
- Open-Meteo no requiere API key (límite 10K requests/día).
