"""Tests for municipality and department endpoints."""
from types import SimpleNamespace
from unittest.mock import patch

from app.routers.municipalities import get_municipalities


class _DepartmentQuery:
    def all(self):
        return [("05", "ANTIOQUIA"), ("15", "BOYACÁ")]


class _Session:
    def __init__(self):
        self.query_count = 0

    def query(self, *_args):
        self.query_count += 1
        return _DepartmentQuery()


def test_list_municipalities_loads_departments_once():
    municipalities = [
        SimpleNamespace(
            dane_code="05001",
            name="MEDELLÍN",
            department_dane_code="05",
            lat=6.2442,
            lng=-75.5812,
            altitude=1495,
            avg_temperature=22.5,
            precipitation=1685.0,
        ),
        SimpleNamespace(
            dane_code="15022",
            name="IZA",
            department_dane_code="15",
            lat=5.612,
            lng=-72.981,
            altitude=2538,
            avg_temperature=12.0,
            precipitation=950.0,
        ),
    ]
    session = _Session()

    with patch(
        "app.routers.municipalities.catalog.get_municipalities",
        return_value=municipalities,
    ):
        response = get_municipalities(db=session)

    assert session.query_count == 1
    assert [municipality.department for municipality in response.municipalities] == [
        "ANTIOQUIA",
        "BOYACÁ",
    ]


def test_list_municipalities(client):
    """List endpoint returns municipalities with count."""
    response = client.get("/api/v1/municipalities")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert len(data["municipalities"]) == data["count"]
    muni = data["municipalities"][0]
    assert "id" in muni
    assert "name" in muni
    assert "department" in muni
    assert "department_id" in muni
    assert "dane_code" in muni
    assert "lat" in muni
    assert "lng" in muni


def test_list_municipalities_by_department_id(client):
    """Filtering by department_id returns only municipalities in that department."""
    response = client.get("/api/v1/municipalities", params={"department_id": "05"})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    for muni in data["municipalities"]:
        assert muni["department_id"] == "05"


def test_get_municipality_by_id(client):
    """Get municipality by DANE code returns full details."""
    response = client.get("/api/v1/municipalities/05001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "05001"
    assert data["dane_code"] == "05001"
    assert data["department_id"] == "05"
    assert data["department"] == "ANTIOQUIA"


def test_get_municipality_not_found(client):
    """Non-existent municipality returns 404."""
    response = client.get("/api/v1/municipalities/99999")
    assert response.status_code == 404


def test_list_departments(client):
    """Departments endpoint returns all departments with DANE codes and counts."""
    response = client.get("/api/v1/municipalities/departments")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 33
    assert len(data["departments"]) == 33
    assert len(data["departments_detailed"]) == 33
    dept = data["departments_detailed"][0]
    assert "dane_code" in dept
    assert "name" in dept
    assert "municipality_count" in dept
    assert dept["municipality_count"] > 0


def test_nearby_municipality(client):
    """Nearby endpoint finds closest municipality to coordinates."""
    # Medellin coordinates
    response = client.get(
        "/api/v1/municipalities/nearby",
        params={"lat": 6.24, "lng": -75.58, "max_distance_km": 50},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "05001"
    assert data["distance_km"] is not None
    assert data["distance_km"] < 50


def test_nearby_outside_colombia(client):
    """Coordinates outside Colombia return 400."""
    response = client.get(
        "/api/v1/municipalities/nearby",
        params={"lat": 50.0, "lng": 0.0},
    )
    assert response.status_code == 400
