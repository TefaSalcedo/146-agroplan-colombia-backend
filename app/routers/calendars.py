from fastapi import APIRouter, Depends, HTTPException, Body, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.schemas.calendar import (
    CalendarBatchRequest,
    CalendarBatchResponse,
    CalendarCropExplanation,
    CalendarCropResult,
    CalendarHarvestWindow,
    CalendarMonthlyForecast,
)
from app.schemas.system import ErrorResponse
from app.services.crop_catalog import CropCatalog
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/calendars", tags=["calendars"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
prediction_service = PredictionService()
logger = get_logger("app.routers.calendars")


@router.post(
    "/predict-batch",
    response_model=CalendarBatchResponse,
    summary="Predict planting calendars for multiple crops",
    description=(
        "Builds 12-month planting calendar predictions for multiple crops in a municipality.\n\n"
        "Uses the XGBoost/LightGBM yield ensemble when yield models are loaded. "
        "Combines EVA, EcoCrop, FAO and yield signals. Returns top 3 harvest months "
        "with computed planting windows. Climate data uses Open-Meteo forecast when available; "
        "otherwise reports missing values with source tracking.\n\n"
        "Each crop result includes its own LLM-generated explanation and token usage audit."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        },
    },
)
def predict_calendar_batch(
    request: CalendarBatchRequest = Body(
        ...,
        examples={
            "medellin_batch": {
                "summary": "Batch calendar for Medellin",
                "value": {
                    "municipality_id": "05001",
                    "crop_ids": ["aguacate", "pina"],
                    "horizon_months": 12,
                },
            }
        },
    ),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] POST /calendars/predict-batch called (municipality_id=%s, crop_ids=%s, horizon_months=%s)", request.municipality_id, request.crop_ids, request.horizon_months)

    logger.debug("[endpoint] Querying database for municipality_id=%s", request.municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", request.municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop_ids = request.crop_ids
    if not crop_ids:
        logger.debug("[endpoint] No crop_ids provided; fetching all ML-supported crops")
        ml_crops = crop_catalog.get_ml_supported_crops(db)
        crop_ids = [c.id for c in ml_crops]
        logger.debug("[endpoint] Predicting calendars for %s crops", len(crop_ids))

    logger.info("[endpoint] Calling PredictionService.predict_calendar_batch")
    raw = prediction_service.predict_calendar_batch(
        db=db,
        municipality_id=request.municipality_id,
        crop_ids=crop_ids,
        horizon_months=request.horizon_months,
        municipality_name=municipality.name,
    )

    results = []
    for r in raw.get("results", []):
        monthly = [CalendarMonthlyForecast(**mf) for mf in r.get("monthly_forecasts", [])]
        harvest = [CalendarHarvestWindow(**hw) for hw in r.get("top_harvest_months", [])]
        explanation = CalendarCropExplanation(**r.get("explanation", {"status": "llm_unavailable"}))
        results.append(
            CalendarCropResult(
                crop_id=r["crop_id"],
                crop_name=r["crop_name"],
                yield_prediction=r.get("yield_prediction"),
                yield_model_version=r.get("yield_model_version"),
                yield_confidence=r.get("yield_confidence"),
                top_harvest_months=harvest,
                monthly_forecasts=monthly,
                warnings=r.get("warnings", []),
                method=r.get("method", "unavailable"),
                explanation=explanation,
            )
        )

    logger.info("[endpoint] POST /calendars/predict-batch returning %s crop results", len(results))
    return CalendarBatchResponse(
        municipality_id=request.municipality_id,
        municipality_name=municipality.name,
        horizon_months=request.horizon_months,
        results=results,
        model_version=raw.get("model_version", "unavailable"),
    )
