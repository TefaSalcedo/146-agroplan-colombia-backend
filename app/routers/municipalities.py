from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Department, Municipality
from app.services.municipality_catalog import MunicipalityCatalog
from app.schemas.municipality import (
    MunicipalityResponse,
    MunicipalityListResponse,
    DepartmentListResponse,
    DepartmentResponse,
)
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/municipalities", tags=["municipalities"])
catalog = MunicipalityCatalog()


def _municipality_to_response(muni: Municipality, db: Session) -> MunicipalityResponse:
    """Convert a Municipality ORM object to a response, joining department name."""
    dept = db.query(Department).filter(Department.dane_code == muni.department_dane_code).first()
    dept_name = dept.name if dept else ""
    return MunicipalityResponse(
        id=muni.dane_code,
        name=muni.name,
        department=dept_name,
        department_id=muni.department_dane_code,
        lat=muni.lat,
        lng=muni.lng,
        altitude=muni.altitude or 0,
        avg_temperature=muni.avg_temperature,
        precipitation=muni.precipitation,
        dane_code=muni.dane_code,
    )


@router.get(
    "",
    response_model=MunicipalityListResponse,
    summary="List municipalities",
    description=(
        "Returns all municipalities covered by AgroPlan.\n\n"
        "Use cases:\n"
        "- Populate municipality selectors in frontend forms.\n"
        "- Filter the catalog by department name or DANE code for localized workflows."
    ),
)
def get_municipalities(
    department: str | None = Query(None, description="Filter by department name (partial match)"),
    department_id: str | None = Query(None, description="Filter by department DANE code (2 digits, exact match)"),
    db: Session = Depends(get_db),
):
    municipalities = catalog.get_municipalities(db, department=department, department_id=department_id)
    responses = [_municipality_to_response(m, db) for m in municipalities]
    return MunicipalityListResponse(municipalities=responses, count=len(responses))


@router.get(
    "/departments",
    response_model=DepartmentListResponse,
    summary="List available departments",
    description=(
        "Returns all departments with DANE codes and municipality counts.\n\n"
        "Use cases:\n"
        "- Build department dropdowns before municipality selection.\n"
        "- Validate if a department is represented in loaded data."
    ),
)
def get_departments(db: Session = Depends(get_db)):
    departments = catalog.get_departments(db)
    dept_responses = [
        DepartmentResponse(
            dane_code=d["dane_code"],
            name=d["name"],
            municipality_count=d["municipality_count"],
        )
        for d in departments
    ]
    return DepartmentListResponse(
        departments=[d.name for d in dept_responses],
        departments_detailed=dept_responses,
        count=len(dept_responses),
    )


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

    response = _municipality_to_response(municipality, db)
    response.distance_km = round(distance_km or 0, 2)
    return response


@router.get(
    "/{municipality_id}",
    response_model=MunicipalityResponse,
    summary="Get municipality by ID",
    description=(
        "Returns one municipality by its DANE code (5 digits).\n\n"
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
    municipality_id: str = Path(..., description="Municipality DANE code (5 digits, e.g. 05001)"),
    db: Session = Depends(get_db),
):
    municipality = catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    return _municipality_to_response(municipality, db)
