import math
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Float, Integer
from app.database import Base


class Municipality(Base):
    __tablename__ = "municipalities"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False, index=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    altitude = Column(Integer, nullable=True)
    avg_temperature = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    dane_code = Column(String, unique=True, index=True)


class MunicipalityCatalog:
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
    
    def get_nearest_municipality(self, db: Session, lat: float, lng: float):
        """Find the nearest municipality to given coordinates using Haversine formula"""
        municipalities = db.query(Municipality).all()
        
        if not municipalities:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for municipality in municipalities:
            distance = self.haversine_distance(lat, lng, municipality.lat, municipality.lng)
            if distance < min_distance:
                min_distance = distance
                nearest = municipality
        
        return nearest
    
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
