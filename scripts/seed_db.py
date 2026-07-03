import csv
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base
from app.config import get_settings

settings = get_settings()


def create_municipality_table():
    """Create municipalities table if it doesn't exist"""
    from sqlalchemy import Column, String, Float, Integer, text

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

    Base.metadata.create_all(bind=engine)
    return Municipality


def seed_municipalities():
    """Seed municipalities from CSV file"""
    Municipality = create_municipality_table()
    
    db: Session = SessionLocal()
    
    try:
        # Check if data already exists
        if db.query(Municipality).count() > 0:
            print("Municipalities table already has data. Skipping seed.")
            return

        csv_path = os.path.join(settings.ml_data_path, "../data/divipola_municipios.csv")
        # Fallback to data/ directory if ml_data_path not set
        if not os.path.exists(csv_path):
            csv_path = "data/divipola_municipios.csv"
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                municipality = Municipality(
                    id=row["id"],
                    name=row["name"],
                    department=row["department"],
                    lat=float(row["lat"]),
                    lng=float(row["lng"]),
                    altitude=int(row["altitude"]),
                    avg_temperature=float(row["avg_temperature"]) if row["avg_temperature"] else None,
                    precipitation=float(row["precipitation"]) if row["precipitation"] else None,
                    dane_code=row["dane_code"],
                )
                db.add(municipality)
        
        db.commit()
        print(f"Seeded {db.query(Municipality).count()} municipalities")
    
    except Exception as e:
        db.rollback()
        print(f"Error seeding municipalities: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_municipalities()
