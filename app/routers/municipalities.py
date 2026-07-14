from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.models import Department, Municipality
from app.services.municipality_catalog import MunicipalityCatalog
from app.schemas.municipality import (
    MunicipalityResponse,
    MunicipalityListResponse,
    DepartmentListResponse,
    DepartmentResponse,
    MunicipalitySearchResult,
    MunicipalitySearchResponse,
)
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/municipalities", tags=["municipalities"])
catalog = MunicipalityCatalog()
logger = get_logger("app.routers.municipalities")


def _department_names(db: Session) -> dict[str, str]:
    return dict(db.query(Department.dane_code, Department.name).all())


def _municipality_to_response(muni: Municipality, department_name: str = "") -> MunicipalityResponse:
    """Convert a Municipality ORM object to a response."""
    return MunicipalityResponse(
        id=muni.dane_code,
        name=muni.name,
        department=department_name,
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
    logger.info("[endpoint] GET /municipalities called (department=%s, department_id=%s)", department, department_id)
    logger.debug("[endpoint] Querying database for municipalities")
    municipalities = catalog.get_municipalities(db, department=department, department_id=department_id)
    logger.debug("[endpoint] Found %s municipalities", len(municipalities))
    department_names = _department_names(db)
    responses = [
        _municipality_to_response(m, department_names.get(m.department_dane_code, ""))
        for m in municipalities
    ]
    logger.info("[endpoint] GET /municipalities returning %s results", len(responses))
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
    logger.info("[endpoint] GET /municipalities/departments called")
    logger.debug("[endpoint] Querying database for departments")
    departments = catalog.get_departments(db)
    logger.debug("[endpoint] Found %s departments", len(departments))
    dept_responses = [
        DepartmentResponse(
            dane_code=d["dane_code"],
            name=d["name"],
            municipality_count=d["municipality_count"],
        )
        for d in departments
    ]
    logger.info("[endpoint] GET /municipalities/departments returning %s results", len(dept_responses))
    return DepartmentListResponse(
        departments=[d.name for d in dept_responses],
        departments_detailed=dept_responses,
        count=len(dept_responses),
    )


@router.get(
    "/search",
    response_model=MunicipalitySearchResponse,
    summary="Search municipalities and departments by name",
    description=(
        "Autocomplete endpoint for the new single selector.\n\n"
        "Returns both municipalities and departments matching the query. "
        "The frontend can use the `type` field to render the appropriate option.\n\n"
        "Use cases:\n"
        "- User types 2+ letters and selects a municipality or department.\n"
        "- Department selections can be expanded to their municipalities on the frontend."
    ),
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "Query must be at least 2 characters.",
        }
    },
)
def search_municipalities(
    q: str = Query(..., description="Search text (minimum 2 characters)", min_length=2, max_length=100),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /municipalities/search called (q=%s, limit=%s)", q, limit)
    logger.debug("[endpoint] Querying database for search query='%s'", q)
    results = catalog.search_municipalities_and_departments(db, q, limit)
    logger.debug("[endpoint] Search returned %s results", len(results))
    logger.info("[endpoint] GET /municipalities/search returning %s results", len(results))
    return MunicipalitySearchResponse(
        query=q,
        results=[MunicipalitySearchResult(**r) for r in results],
        count=len(results),
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
    logger.info("[endpoint] GET /municipalities/nearby called (lat=%s, lng=%s, max_distance_km=%s)", lat, lng, max_distance_km)
    logger.debug("[endpoint] Querying database for nearest municipality")
    if not catalog.is_within_colombia(lat, lng):
        logger.warning("[endpoint] Coordinates outside Colombia (lat=%s, lng=%s)", lat, lng)
        raise HTTPException(status_code=400, detail="Coordinates are outside Colombia")

    municipality, distance_km = catalog.get_nearest_municipality(db, lat, lng, max_distance_km)
    if not municipality:
        detail = "No covered municipality was found within the allowed distance"
        if distance_km is not None:
            detail = f"Nearest municipality is {distance_km:.1f} km away, outside the allowed distance"
        logger.warning("[endpoint] No nearby municipality found: %s", detail)
        raise HTTPException(status_code=404, detail=detail)

    response = _municipality_to_response(
        municipality,
        _department_names(db).get(municipality.department_dane_code, ""),
    )
    response.distance_km = round(distance_km or 0, 2)
    logger.info("[endpoint] GET /municipalities/nearby returning municipality_id=%s distance_km=%s", response.id, response.distance_km)
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
    logger.info("[endpoint] GET /municipalities/{municipality_id} called (municipality_id=%s)", municipality_id)
    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")
    logger.info("[endpoint] GET /municipalities/{municipality_id} returning municipality_id=%s", municipality_id)
    return _municipality_to_response(
        municipality,
        _department_names(db).get(municipality.department_dane_code, ""),
    )
