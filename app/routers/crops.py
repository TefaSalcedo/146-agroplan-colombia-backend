from fastapi import APIRouter, HTTPException
from app.services.crop_catalog import CropCatalog
from app.schemas.crop import CropResponse, CropListResponse, CropResponseLite

router = APIRouter(prefix="/crops", tags=["crops"])
catalog = CropCatalog()


@router.get("", response_model=CropListResponse)
def get_crops():
    """Get all crops with full details"""
    crops = catalog.get_all_crops()
    return CropListResponse(crops=crops, count=len(crops))


@router.get("/lite", response_model=list[CropResponseLite])
def get_crops_lite():
    """Get all crops with lite version (for lists)"""
    return catalog.get_crops_lite()


@router.get("/{crop_id}", response_model=CropResponse)
def get_crop(crop_id: str):
    """Get a crop by ID"""
    crop = catalog.get_crop_by_id(crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    return crop
