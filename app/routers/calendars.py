from fastapi import APIRouter, Depends, HTTPException, Body, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.crop_catalog import CropCatalog
from app.services.mock_predictor import MockPredictor
from app.schemas.calendar import CalendarRequest, CalendarResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/calendars", tags=["calendars"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
predictor = MockPredictor()


@router.post(
    "/predict",
    response_model=CalendarResponse,
    summary="Predict planting calendar",
    description=(
        "Builds day-level planting ratings for a specific crop and municipality month.\n\n"
        "Use cases:\n"
        "- Visual calendar heatmaps in planning UX.\n"
        "- Compare months before scheduling planting activities."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop or municipality ID not found.",
        }
    },
)
def predict_calendar(
    request: CalendarRequest = Body(
        ...,
        examples={
            "corn_august": {
                "summary": "Corn plan for August 2026",
                "value": {
                    "crop_id": "maiz",
                    "municipality_id": "17001",
                    "month": 8,
                    "year": 2026,
                },
            }
        },
    ),
    db: Session = Depends(get_db),
):
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop = crop_catalog.get_crop_by_id(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    prediction = predictor.predict_calendar(
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
        month=request.month,
        year=request.year,
        planting_months=crop.planting_months,
    )

    return CalendarResponse(**prediction)
