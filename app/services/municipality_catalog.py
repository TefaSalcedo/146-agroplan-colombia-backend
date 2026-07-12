import math
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Municipality, Department


class MunicipalityCatalog:
    COLOMBIA_LAT_MIN = -4.5
    COLOMBIA_LAT_MAX = 13.6
    COLOMBIA_LNG_MIN = -79.1
    COLOMBIA_LNG_MAX = -66.8

    def get_municipalities(
        self,
        db: Session,
        department: Optional[str] = None,
        department_id: Optional[str] = None,
    ):
        """Get all municipalities, optionally filtered by department name or DANE code."""
        query = db.query(Municipality)

        if department_id:
            query = query.filter(Municipality.department_dane_code == department_id)
        elif department:
            # Join with departments for name-based filtering (backwards compatible)
            query = query.join(
                Department, Municipality.department_dane_code == Department.dane_code
            ).filter(Department.name.ilike(f"%{department}%"))

        return query.order_by(Municipality.name).all()

    def get_municipality_by_id(self, db: Session, municipality_id: str):
        """Get a municipality by its DANE code (5-digit)."""
        return db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()

    def get_departments(self, db: Session) -> list[dict]:
        """Get all departments with municipality counts."""
        results = (
            db.query(
                Department.dane_code,
                Department.name,
                func.count(Municipality.dane_code).label("municipality_count"),
            )
            .outerjoin(Municipality, Municipality.department_dane_code == Department.dane_code)
            .group_by(Department.dane_code, Department.name)
            .order_by(Department.name)
            .all()
        )
        return [
            {"dane_code": r.dane_code, "name": r.name, "municipality_count": r.municipality_count}
            for r in results
        ]

    def get_department_names(self, db: Session) -> list[str]:
        """Get all unique department names (backwards compatible)."""
        results = db.query(Department.name).order_by(Department.name).all()
        return [r[0] for r in results]

    def search_municipalities_and_departments(self, db: Session, query: str, limit: int = 20):
        """Search municipalities and departments by partial name match.

        Returns mixed results for autocomplete inputs where users type either a
        municipality or a department name. Minimum 2 characters are expected.
        """
        normalized = query.strip().lower()
        if len(normalized) < 2:
            return []

        # Search departments
        departments = (
            db.query(Department)
            .filter(Department.name.ilike(f"%{normalized}%"))
            .order_by(Department.name)
            .limit(limit)
            .all()
        )

        # Search municipalities, joining department for name
        municipalities = (
            db.query(Municipality, Department)
            .join(Department, Municipality.department_dane_code == Department.dane_code)
            .filter(Municipality.name.ilike(f"%{normalized}%"))
            .order_by(Municipality.name)
            .limit(limit)
            .all()
        )

        results = []
        for dept in departments:
            results.append({
                "id": dept.dane_code,
                "name": dept.name,
                "type": "department",
                "department_id": None,
                "department_name": None,
            })

        for muni, dept in municipalities:
            results.append({
                "id": muni.dane_code,
                "name": muni.name,
                "type": "municipality",
                "department_id": dept.dane_code,
                "department_name": dept.name,
            })

        # Prioritize exact prefix matches, then by name length
        results.sort(key=lambda r: (
            0 if r["name"].lower().startswith(normalized) else 1,
            len(r["name"]),
            r["name"].lower(),
        ))

        return results[:limit]

    def get_nearest_municipality(self, db: Session, lat: float, lng: float, max_distance_km: float = 20):
        """Find the nearest covered municipality to given coordinates using Haversine formula."""
        if not self.is_within_colombia(lat, lng):
            return None, None

        municipalities = db.query(Municipality).all()

        if not municipalities:
            return None, None

        nearest = None
        min_distance = float("inf")

        for municipality in municipalities:
            distance = self.haversine_distance(lat, lng, municipality.lat, municipality.lng)
            if distance < min_distance:
                min_distance = distance
                nearest = municipality

        if nearest is None or min_distance > max_distance_km:
            return None, min_distance if nearest else None

        return nearest, min_distance

    @classmethod
    def is_within_colombia(cls, lat: float, lng: float) -> bool:
        return (
            cls.COLOMBIA_LAT_MIN <= lat <= cls.COLOMBIA_LAT_MAX
            and cls.COLOMBIA_LNG_MIN <= lng <= cls.COLOMBIA_LNG_MAX
        )

    @staticmethod
    def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula (in kilometers)."""
        R = 6371

        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)

        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
