"""Tests for national crop guide persistence."""
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy.dialects import postgresql

from app.services.crop_national_guide_service import CropNationalGuideService


class _Result:
    def __init__(self, guide):
        self.guide = guide

    def scalar_one(self):
        return self.guide


class _Session:
    def __init__(self, guide):
        self.guide = guide
        self.executed_statement = None
        self.committed = False
        self.rolled_back = False

    def query(self, *_args):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return None

    def execute(self, statement):
        self.executed_statement = statement
        return _Result(self.guide)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_generation_persists_national_guide_with_atomic_upsert():
    crop = SimpleNamespace(
        id="pina",
        name="Piña",
        scientific_name="Ananas comosus",
        days_to_harvest=365,
        establishment_period_days=365,
        is_perennial=True,
        soil_type="Franco",
        ideal_temperature="20-30 C",
        humidity="70%",
        precipitation="1000 mm",
        altitude="0-1000 msnm",
        irrigation="Moderado",
        substrates=[],
        planting_months=[],
        harvest_months=[],
        tips=[],
    )
    persisted_guide = SimpleNamespace(
        content='{"summary": "Guía de Piña", "sections": []}',
        generated_at=None,
        expires_at=None,
        provider="test-provider",
        model="test-model",
        tokens_in=12,
        tokens_out=34,
        latency_ms=56,
    )
    session = _Session(persisted_guide)
    service = CropNationalGuideService()
    service.llm_service = SimpleNamespace(
        generate_national_crop_guide=lambda _crop: {
            "status": "success",
            "summary": "Guía de Piña",
            "sections": [],
            "provider": "test-provider",
            "model": "test-model",
            "tokens_in": 12,
            "tokens_out": 34,
            "latency_ms": 56,
        }
    )

    with patch("app.services.crop_national_guide_service.get_settings", return_value=SimpleNamespace(llm_enabled=True)), patch(
        "app.services.crop_national_guide_service.log_llm_generation", return_value=1
    ):
        response = service.get_or_generate(session, crop)

    statement = str(session.executed_statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_crop_national_guide_crop DO UPDATE" in statement
    assert session.rolled_back is True
    assert session.committed is True
    assert response["status"] == "success"
    assert response["summary"] == "Guía de Piña"
