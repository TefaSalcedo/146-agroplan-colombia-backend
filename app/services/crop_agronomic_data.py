"""Verified agronomic facts for the 7 zoning crops.

These values are used only when EVA/FAO datasets do not provide a calendar
field (sowing months or cycle duration). Each value is backed by an external
source to avoid arbitrary hardcoding.

Sources:
- Aguacate: Fenología del aguacate cv. Hass en Antioquia (Corpoica/UNAL 2017)
  https://www.avocadosource.com/Journals/Memorias_VCLA/2017/Memorias_VCLA_2017_PG_292.pdf
  First commercial harvest starts at 30-36 months; main harvest windows
  Feb-May and Sep-Dec. Treated as a permanent crop with annual production cycles.
- Piña: Agrosavia / UNAL practices for Ananas comosus in Colombia
  https://repository.agrosavia.co/server/api/core/bitstreams/16bf32f6-fbc0-4ec2-be79-779d8009de46/content
  https://repository.unad.edu.co/handle/10596/58416
  Fresh-market cycle 16-18 months (Gold Honey), up to 24 months for other varieties.
- Fresa: USB / Agronegocios Colombia
  https://revistas.usb.edu.co/index.php/IngUSBmed/article/download/2564/2493
  https://www.agronegocios.co/agricultura/fresa-un-cultivo-rentable-y-con-proyeccion-en-la-exterio-2621276
  First harvest ~7 months after planting; full productive cycle ~2 years.
  Recommended planting in dry months to reduce fungal pressure.
- Caña panelera: Agrosavia guía técnica panela
  https://doi.org/10.21930/agrosavia.nbook.7408614
  Cycle 12-22 months depending on altitude; first cut 8-12 months.
- Algodón: Conalgodón / ICA Colombia
  https://elcampesino.co/temporada-siembra-algodon/
  https://www.conalgodon.com/el-algodon-el-cultivo-local-de-ciclo-corto-mas-competitivo-en-el-mercado-internacional/
  Cycle 100-140 days; sowing interior Feb-Mar, coast/llanos Aug-Oct.
"""

from typing import Dict, List

_CROP_AGRONOMIC_FACTS: Dict[str, Dict] = {
    "aguacate": {
        "cycle_days_min": 365,
        "cycle_days_max": 365,
        # First commercial harvest is 30-36 months after planting; reported as a
        # range for users who need an establishment horizon.
        "establishment_days_min": 900,
        "establishment_days_max": 1095,
        "planting_months": [2, 3, 4, 5, 9, 10, 11, 12],
        "harvest_months": [2, 3, 4, 5, 9, 10, 11, 12],
        "perennial": True,
        "sources": [
            "https://www.avocadosource.com/Journals/Memorias_VCLA/2017/Memorias_VCLA_2017_PG_292.pdf",
            "https://repository.udistrital.edu.co/bitstreams/c5df2ee0-63d7-4735-9b9e-12b5f6132465/download",
        ],
    },
    "pina": {
        "cycle_days_min": 480,
        "cycle_days_max": 730,
        "planting_months": [3, 4, 5, 6],
        "harvest_months": [7, 8, 9, 10, 11, 12],
        "perennial": False,
        "sources": [
            "https://repository.agrosavia.co/server/api/core/bitstreams/16bf32f6-fbc0-4ec2-be79-779d8009de46/content",
            "https://repository.unad.edu.co/handle/10596/58416",
            "https://www.agronegocios.co/agricultura/la-dosis-adecuada-de-fertilizantes-mejoraria-la-calidad-nutritiva-de-la-pina-oro-miel-3601627",
        ],
    },
    "fresa": {
        "cycle_days_min": 180,
        "cycle_days_max": 210,
        # Full commercial cycle can be 2 years, but the first productive cycle is
        # what matters for the calendar endpoint.
        "productive_cycle_days_min": 180,
        "productive_cycle_days_max": 730,
        "planting_months": [3, 4, 9, 10],
        "harvest_months": [11, 12, 1, 2, 3, 4],
        "perennial": False,
        "sources": [
            "https://revistas.usb.edu.co/index.php/IngUSBmed/article/download/2564/2493",
            "https://www.agronegocios.co/agricultura/fresa-un-cultivo-rentable-y-con-proyeccion-en-la-exterio-2621276",
        ],
    },
    "cana_panelera": {
        "cycle_days_min": 240,
        "cycle_days_max": 365,
        "planting_months": [3, 4, 5, 6],
        "harvest_months": [8, 9, 10, 11, 12],
        "perennial": True,
        "sources": [
            "https://doi.org/10.21930/agrosavia.nbook.7408614",
        ],
    },
    "algodon": {
        "cycle_days_min": 100,
        "cycle_days_max": 140,
        "planting_months": [2, 3, 8, 9, 10],
        "harvest_months": [6, 7, 11, 12, 1],
        "perennial": False,
        "sources": [
            "https://elcampesino.co/temporada-siembra-algodon/",
            "https://www.conalgodon.com/el-algodon-el-cultivo-local-de-ciclo-corto-mas-competitivo-en-el-mercado-internacional/",
        ],
    },
}


def get_agronomic_facts(crop_id: str) -> Dict:
    """Return verified agronomic facts for a crop, or an empty dict if unknown."""
    return _CROP_AGRONOMIC_FACTS.get(crop_id, {})


def list_supported_crops() -> List[str]:
    """Return crop ids with verified agronomic facts."""
    return list(_CROP_AGRONOMIC_FACTS.keys())
