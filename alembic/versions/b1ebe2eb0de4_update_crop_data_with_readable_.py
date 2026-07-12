"""Update crop data with readable durations and avocado perennial info

Revision ID: b1ebe2eb0de4
Revises: 76161805c979
Create Date: 2026-07-12 12:14:03.424916

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision: str = 'b1ebe2eb0de4'
down_revision: Union[str, None] = '76161805c979'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update avocado to reflect its perennial nature and normalize crop soil text."""
    bind = op.get_bind()
    session = Session(bind=bind)

    # Avocado is a permanent crop: years to establishment, months between flowering and harvest.
    avocado_tips = [
        {
            "title": "Es un árbol permanente",
            "description": "El aguacate no es un cultivo de un solo ciclo. Después de sembrarlo, tarda varios años en dar cosechas. Una vez adulto, produce frutos varias veces al año."
        },
        {
            "title": "Acodo aéreo para propagar",
            "description": "Se puede reproducir con la técnica de acodo aéreo: se pela un trozo de corteza en una rama sana, se cubre con tierra negra o musgo húmedo, se envuelve en plástico y se espera a que salgan raíces. Después se corta la rama y se siembra como un nuevo árbol, igual a la planta madre."
        },
        {
            "title": "No encharcar",
            "description": "El aguacate necesita agua regular pero sufre mucho si el suelo se encharca. Asegúrate de que la tierra deje salir bien el agua."
        },
        {
            "title": "Proteger de vientos fuertes",
            "description": "Las ramas son quebradizas cuando cargan fruto. Plántalo en sitios protegidos del viento fuerte o use barreras vivas."
        }
    ]

    op.execute(
        sa.text("""
            UPDATE crops
            SET is_perennial = true,
                establishment_period_days = 2190,
                days_to_harvest = 180,
                soil_type = 'Tierra suelta y profunda que drene bien; pH 4.5 - 7.0',
                tips = :tips
            WHERE id = 'aguacate'
        """).bindparams(tips=json.dumps(avocado_tips, ensure_ascii=False))
    )

    # Normalize soil_type strings so the backend simplifier produces clean plain-language text.
    # Avoid repeating an acidity descriptor before the pH range.
    soil_updates = {
        "algodon": "Franco a franco arcilloso, profundo, bien drenado; pH 5.0 - 9.5",
        "cana_panelera": "Franco, profundo, buena materia orgánica; pH 4.5 - 9.0",
        "cebolla": "Franco arenoso, suelto y bien drenado; pH 4.3 - 8.3",
        "fresa": "Franco arenoso, rico en materia orgánica; pH 4.5 - 6.5",
        "pina": "Franco arenoso, ligeramente ácido, bien drenado; pH 3.5 - 9.0",
        "soya": "Franco, bien drenado; pH 5.5 - 7.0",
    }

    for crop_id, soil_type in soil_updates.items():
        op.execute(
            sa.text("UPDATE crops SET soil_type = :soil_type WHERE id = :crop_id")
            .bindparams(crop_id=crop_id, soil_type=soil_type)
        )

    session.commit()


def downgrade() -> None:
    """Restore previous avocado values and soil text."""
    bind = op.get_bind()
    session = Session(bind=bind)

    op.execute(
        sa.text("""
            UPDATE crops
            SET is_perennial = false,
                establishment_period_days = NULL,
                days_to_harvest = 365,
                soil_type = 'Franco arenoso, profundo, bien drenado; pH 4.5 - 7.0',
                tips = NULL
            WHERE id = 'aguacate'
        """)
    )

    previous_soil = {
        "algodon": "Franco a franco arcilloso, profundo, bien drenado; pH 5.0 - 9.5",
        "cana_panelera": "Franco, profundo, buena materia organica; pH 4.5 - 9.0",
        "cebolla": "Franco arenoso, suelto y bien drenado; pH 4.3 - 8.3",
        "fresa": "Franco arenoso, rico en materia organica; pH 4.5 - 6.5",
        "pina": "Franco arenoso, ligeramente acido, bien drenado; pH 3.5 - 9.0",
        "soya": "Franco, bien drenado; pH 5.5 - 7.0",
    }

    for crop_id, soil_type in previous_soil.items():
        op.execute(
            sa.text("UPDATE crops SET soil_type = :soil_type WHERE id = :crop_id")
            .bindparams(crop_id=crop_id, soil_type=soil_type)
        )

    session.commit()
