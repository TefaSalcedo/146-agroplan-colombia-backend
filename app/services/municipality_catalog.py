import math
from sqlalchemy.orm import Session
from app.models import Municipality


class MunicipalityCatalog:
    COLOMBIA_LAT_MIN = -4.5
    COLOMBIA_LAT_MAX = 13.6
    COLOMBIA_LNG_MIN = -79.1
    COLOMBIA_LNG_MAX = -66.8

    def get_municipalities(self, db: Session, department: str | None = None):
        """Get all municipalities, optionally filtered by department"""
        query = db.query(Municipality)
        
        if department:
            query = query.filter(Municipality.department.ilike(f"%{department}%"))
        
        municipalities = query.order_by(Municipality.name).all()
        return municipalities
    
    def get_municipality_by_id(self, db: Session, municipality_id: str):
        """Get a municipality by ID"""
        return db.query(Municipality).filter(Municipality.id == municipality_id).first()
    
    def get_departments(self, db: Session):
        """Get all unique departments"""
        departments = db.query(Municipality.department).distinct().order_by(Municipality.department).all()
        return [d[0] for d in departments]
    
    def get_nearest_municipality(self, db: Session, lat: float, lng: float, max_distance_km: float = 20):
        """Find the nearest covered municipality to given coordinates using Haversine formula"""
        if not self.is_within_colombia(lat, lng):
            return None, None

        municipalities = db.query(Municipality).all()
        
        if not municipalities:
            return None, None
        
        nearest = None
        min_distance = float('inf')
        
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
        """Calculate distance between two points using Haversine formula (in kilometers)"""
        # Earth radius in kilometers
        R = 6371
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
