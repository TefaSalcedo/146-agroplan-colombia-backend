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
    altitude = Column(Integer, nullable=False)
    avg_temperature = Column(Float)
    precipitation = Column(Float)
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
