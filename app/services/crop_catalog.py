from typing import List, Optional

from sqlalchemy.orm import Session

from app.models import Crop
from app.schemas.crop import CropResponse, CropResponseLite, GrowthStage, Tip
from app.utils.crop_formatting import (
    format_duration_days,
    simplify_soil_terms,
    simplify_temperature,
    simplify_humidity,
    simplify_precipitation,
    simplify_altitude,
)


class CropCatalog:
    """Crop catalog backed by the database.

    Only the 7 ML-supported crops are stored. Agronomic fields come from
    traced data sources; no mock text is silently reused.
    """

    def _to_response(self, crop: Crop) -> CropResponse:
        stages = [GrowthStage(**s) for s in (crop.stages or [])]
        tips = [Tip(**t) for t in (crop.tips or [])]
        return CropResponse(
            id=crop.id,
            name=crop.name,
            scientific_name=crop.scientific_name or "",
            image=crop.image or "",
            days_to_harvest=crop.days_to_harvest or 0,
            days_to_harvest_text=format_duration_days(crop.days_to_harvest),
            establishment_period_days=crop.establishment_period_days,
            establishment_period_text=format_duration_days(crop.establishment_period_days),
            is_perennial=crop.is_perennial or False,
            soil_type=crop.soil_type or "",
            soil_type_simple=simplify_soil_terms(crop.soil_type or ""),
            ideal_temperature=crop.ideal_temperature or "",
            ideal_temperature_simple=simplify_temperature(crop.ideal_temperature or ""),
            humidity=crop.humidity or "",
            humidity_simple=simplify_humidity(crop.humidity or ""),
            precipitation=crop.precipitation or "",
            precipitation_simple=simplify_precipitation(crop.precipitation or ""),
            altitude=crop.altitude or "",
            altitude_simple=simplify_altitude(crop.altitude or ""),
            irrigation=crop.irrigation or "",
            substrates=crop.substrates or [],
            planting_months=crop.planting_months or [],
            harvest_months=crop.harvest_months or [],
            stages=stages,
            tips=tips,
        )

    def _to_lite(self, crop: Crop) -> CropResponseLite:
        return CropResponseLite(
            id=crop.id,
            name=crop.name,
            image=crop.image or "",
        )

    def get_all_crops(self, db: Session) -> List[CropResponse]:
        crops = db.query(Crop).order_by(Crop.name).all()
        return [self._to_response(c) for c in crops]

    def get_crops_lite(self, db: Session) -> List[CropResponseLite]:
        crops = db.query(Crop).order_by(Crop.name).all()
        return [self._to_lite(c) for c in crops]

    def get_crop_by_id(self, db: Session, crop_id: str) -> Optional[CropResponse]:
        crop = db.query(Crop).filter(Crop.id == crop_id).first()
        if not crop:
            return None
        return self._to_response(crop)

    def get_crop_model_by_id(self, db: Session, crop_id: str) -> Optional[Crop]:
        """Return the raw ORM model for internal use."""
        return db.query(Crop).filter(Crop.id == crop_id).first()

    def get_ml_supported_crops(self, db: Session) -> List[Crop]:
        """Return only crops that have ML models trained."""
        return db.query(Crop).filter(Crop.is_ml_supported == True).order_by(Crop.name).all()
