from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.schemas.municipality import MunicipalityResponse, MunicipalityListResponse

router = APIRouter(prefix="/municipalities", tags=["municipalities"])
catalog = MunicipalityCatalog()


@router.get("", response_model=MunicipalityListResponse)
def get_municipalities(
    department: str | None = Query(None, description="Filter by department"),
    db: Session = Depends(get_db)
):
    """Get all municipalities, optionally filtered by department"""
    municipalities = catalog.get_municipalities(db, department)
    return MunicipalityListResponse(
        municipalities=municipalities,
        count=len(municipalities)
    )


@router.get("/departments")
def get_departments(db: Session = Depends(get_db)):
    """Get all unique departments"""
    departments = catalog.get_departments(db)
    return {"departments": departments}


@router.get("/nearby", response_model=MunicipalityResponse)
def get_nearby_municipality(
    lat: float = Query(..., description="Latitude coordinate"),
    lng: float = Query(..., description="Longitude coordinate"),
    db: Session = Depends(get_db)
):
    """Get the nearest municipality to given coordinates"""
    municipality = catalog.get_nearest_municipality(db, lat, lng)
    if not municipality:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No municipalities found")
    return municipality


@router.get("/{municipality_id}", response_model=MunicipalityResponse)
def get_municipality(municipality_id: str, db: Session = Depends(get_db)):
    """Get a municipality by ID"""
    municipality = catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Municipality not found")
    return municipality
