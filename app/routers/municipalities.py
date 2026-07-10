from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.schemas.municipality import MunicipalityResponse, MunicipalityListResponse, DepartmentListResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/municipalities", tags=["municipalities"])
catalog = MunicipalityCatalog()


@router.get(
    "",
    response_model=MunicipalityListResponse,
    summary="List municipalities",
    description=(
        "Returns all municipalities covered by AgroPlan.\n\n"
        "Use cases:\n"
        "- Populate municipality selectors in frontend forms.\n"
        "- Filter the catalog by department for localized workflows."
    ),
)
def get_municipalities(
    department: str | None = Query(None, description="Optional department filter (exact match)"),
    db: Session = Depends(get_db),
):
    municipalities = catalog.get_municipalities(db, department)
    return MunicipalityListResponse(
        municipalities=municipalities,
        count=len(municipalities),
    )


@router.get(
    "/departments",
    response_model=DepartmentListResponse,
    summary="List available departments",
    description=(
        "Returns unique department names available in the municipalities dataset.\n\n"
        "Use cases:\n"
        "- Build department dropdowns before municipality selection.\n"
        "- Validate if a department is represented in loaded data."
    ),
)
def get_departments(db: Session = Depends(get_db)):
    departments = catalog.get_departments(db)
    return DepartmentListResponse(departments=departments)


@router.get(
    "/nearby",
    response_model=MunicipalityResponse,
    summary="Find nearest municipality",
    description=(
        "Finds the nearest covered municipality from a latitude/longitude pair.\n\n"
        "Use cases:\n"
        "- GPS-first UX to auto-select municipality.\n"
        "- Validate if a user location is covered by current dataset."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Coordinates are outside Colombia.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No municipality found inside the requested distance.",
        },
    },
)
def get_nearby_municipality(
    lat: float = Query(..., ge=-90, le=90, description="Latitude coordinate"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude coordinate"),
    max_distance_km: float = Query(20, gt=0, le=100, description="Maximum allowed distance in kilometers"),
    db: Session = Depends(get_db),
):
    if not catalog.is_within_colombia(lat, lng):
        raise HTTPException(status_code=400, detail="Coordinates are outside Colombia")

    municipality, distance_km = catalog.get_nearest_municipality(db, lat, lng, max_distance_km)
    if not municipality:
        detail = "No covered municipality was found within the allowed distance"
        if distance_km is not None:
            detail = f"Nearest municipality is {distance_km:.1f} km away, outside the allowed distance"
        raise HTTPException(status_code=404, detail=detail)

    municipality.distance_km = round(distance_km or 0, 2)
    return municipality


@router.get(
    "/{municipality_id}",
    response_model=MunicipalityResponse,
    summary="Get municipality by ID",
    description=(
        "Returns one municipality by its AgroPlan identifier.\n\n"
        "Use cases:\n"
        "- Resolve user selections from stored IDs.\n"
        "- Fetch municipality metadata before prediction calls."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID does not exist in catalog.",
        }
    },
)
def get_municipality(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    db: Session = Depends(get_db),
):
    municipality = catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    return municipality
