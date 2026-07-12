from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.services.crop_catalog import CropCatalog
from app.schemas.crop import CropResponse, CropListResponse, CropResponseLite
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/crops", tags=["crops"])
catalog = CropCatalog()
logger = get_logger("app.routers.crops")


@router.get(
    "",
    response_model=CropListResponse,
    summary="List full crop catalog",
    description=(
        "Returns all ML-supported crops with full agronomic details.\n\n"
        "Use cases:\n"
        "- Detailed crop profile pages.\n"
        "- Inputs for recommendation and planning interfaces."
    ),
)
def get_crops(db: Session = Depends(get_db)):
    logger.info("[endpoint] GET /crops called")
    logger.debug("[endpoint] Querying database for all crops")
    crops = catalog.get_all_crops(db)
    logger.info("[endpoint] GET /crops returning %s crops", len(crops))
    return CropListResponse(crops=crops, count=len(crops))


@router.get(
    "/lite",
    response_model=list[CropResponseLite],
    summary="List lite crop cards",
    description=(
        "Returns lightweight crop entries intended for quick lists/cards.\n\n"
        "Use cases:\n"
        "- Fast-loading crop pickers.\n"
        "- Recommendation side panels with minimal payload."
    ),
)
def get_crops_lite(db: Session = Depends(get_db)):
    logger.info("[endpoint] GET /crops/lite called")
    logger.debug("[endpoint] Querying database for lite crop list")
    crops = catalog.get_crops_lite(db)
    logger.info("[endpoint] GET /crops/lite returning %s crops", len(crops))
    return crops


@router.get(
    "/{crop_id}",
    response_model=CropResponse,
    summary="Get crop by ID",
    description=(
        "Returns one crop with all agronomic fields and guidance text.\n\n"
        "Use cases:\n"
        "- Fetch details for a selected crop.\n"
        "- Validate a crop identifier before prediction workflows."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop ID does not exist in catalog.",
        }
    },
)
def get_crop(
    crop_id: str = Path(..., description="Crop identifier (e.g. aguacate, pina)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /crops/{crop_id} called (crop_id=%s)", crop_id)
    logger.debug("[endpoint] Querying database for crop_id=%s", crop_id)
    crop = catalog.get_crop_by_id(db, crop_id)
    if not crop:
        logger.warning("[endpoint] Crop not found: %s", crop_id)
        raise HTTPException(status_code=404, detail="Crop not found")
    logger.info("[endpoint] GET /crops/{crop_id} returning crop_id=%s", crop_id)
    return crop
