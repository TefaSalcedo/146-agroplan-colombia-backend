from fastapi import APIRouter, HTTPException, Path, status
from app.services.crop_catalog import CropCatalog
from app.schemas.crop import CropResponse, CropListResponse, CropResponseLite
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/crops", tags=["crops"])
catalog = CropCatalog()


@router.get(
    "",
    response_model=CropListResponse,
    summary="List full crop catalog",
    description=(
        "Returns all crops with full agronomic details and educational content.\n\n"
        "Use cases:\n"
        "- Detailed crop profile pages.\n"
        "- Inputs for recommendation and planning interfaces."
    ),
)
def get_crops():
    crops = catalog.get_all_crops()
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
def get_crops_lite():
    return catalog.get_crops_lite()


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
def get_crop(crop_id: str = Path(..., description="Crop identifier (for example: cafe, maiz)")):
    crop = catalog.get_crop_by_id(crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    return crop
