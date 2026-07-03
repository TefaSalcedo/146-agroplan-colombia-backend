import json
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
    from sqlalchemy import Column, String, Float, Integer

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

    Base.metadata.create_all(bind=engine)
    return Municipality


def convert_coordinate(coord_str: str) -> float:
    """Convert coordinate from comma decimal to point decimal format"""
    return float(coord_str.replace(",", "."))


def seed_municipalities(force=False):
    """Seed municipalities from JSON file"""
    Municipality = create_municipality_table()
    
    db: Session = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = db.query(Municipality).count()
        if existing_count > 0 and not force:
            print(f"Municipalities table already has {existing_count} records. Skipping seed.")
            return
        
        # If force, delete existing data
        if force and existing_count > 0:
            print(f"Deleting {existing_count} existing municipality records...")
            db.query(Municipality).delete()
            db.commit()

        # Try data/ directory first
        json_path = "data/divipola_municipios.json"
        if not os.path.exists(json_path):
            json_path = os.path.join(settings.ml_data_path, "../data/divipola_municipios.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found at {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            for row in data:
                # Generate ID using department code + municipality code
                municipality_id = f"{row['cod_dpto']}{row['cod_mpio']}"
                
                municipality = Municipality(
                    id=municipality_id,
                    name=row["nom_mpio"],
                    department=row["dpto"],
                    lat=convert_coordinate(row["latitud"]),
                    lng=convert_coordinate(row["longitud"]),
                    altitude=0,  # Default value - not available in DIVIPOLA data
                    avg_temperature=0.0,  # Default value - not available in DIVIPOLA data
                    precipitation=0.0,  # Default value - not available in DIVIPOLA data
                    dane_code=f"{row['cod_dpto']}{row['cod_mpio']}",
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
    force = "--force" in sys.argv
    seed_municipalities(force=force)
