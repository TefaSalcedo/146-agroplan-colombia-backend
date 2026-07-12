"""Feature builders for real ML inference.

Converts backend domain objects (crop_id, municipality DANE code) into the
feature matrices expected by the trained zoning and yield models.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session


# Map backend crop ids to names used in the yield historical profiles.
# Yield profiles use Spanish EVA crop names; zoning uses backend ids directly.
_YIELD_CROP_NAME_MAP = {
    "aguacate": "Aguacate",
    "algodon": "Algodón",
    "cana_panelera": "Caña",
    "cebolla": "Cebolla de bulbo",
    "fresa": "Fresa",
    "pina": "Piña",
    "soya": "Soya",
}

# Map backend crop ids to names used in crop_features_integrated.csv and
# calendar_for_eva.csv (cultivo_eva column).
_EVA_CROP_NAME_MAP = {
    "aguacate": "Aguacate",
    "algodon": "Algodón",
    "cana_panelera": "Caña",
    "cebolla": "Cebolla de bulbo",
    "fresa": "Fresa",
    "pina": "Piña",
    "soya": "Soya",
}


def _get_crop_features_row(loader: Any, crop_id: str) -> Optional[pd.Series]:
    """Return the integrated crop feature row for a backend crop id."""
    features = loader.crop_features
    if features is None:
        return None
    eva_name = _EVA_CROP_NAME_MAP.get(crop_id)
    if not eva_name:
        return None
    rows = features[features["cultivo_eva"] == eva_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _get_calendar_row(loader: Any, crop_id: str) -> Optional[pd.Series]:
    """Return the EVA calendar row for a backend crop id."""
    calendar = loader.calendar_for_eva
    if calendar is None:
        return None
    eva_name = _EVA_CROP_NAME_MAP.get(crop_id)
    if not eva_name:
        return None
    rows = calendar[calendar["cultivo_eva"] == eva_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _get_municipality_profile(loader: Any, municipality_id: str) -> Optional[pd.Series]:
    """Return the municipality profile row for a DANE code."""
    profiles = loader.municipality_profiles
    if profiles is None:
        return None
    try:
        dane_int = int(municipality_id)
    except ValueError:
        return None
    row = profiles[profiles["cod_dane_m"] == dane_int]
    if row.empty:
        return None
    return row.iloc[0]


def _encode_zoning_categoricals(
    df: pd.DataFrame,
    encoder: Any,
) -> pd.DataFrame:
    """Apply the fitted OrdinalEncoder to cultivo and zona_climatica."""
    encoded = df.copy()
    cat_cols = ["cultivo", "zona_climatica"]
    present = [c for c in cat_cols if c in encoded.columns]
    if present:
        encoded[present] = encoder.transform(encoded[present].astype(str))
    return encoded


def build_zoning_features(
    db: Session,
    crop_id: str,
    municipality_id: str,
    loader: Any,
) -> Optional[pd.DataFrame]:
    """Build the 60-feature vector for the zoning LightGBM model.

    Returns None if the municipality profile is missing or the model artifacts
    are not loaded.
    """
    if (
        not loader.is_zoning_model_loaded()
        or loader.zoning_preprocessor is None
        or loader.zoning_feature_schema is None
        or loader.municipality_profiles is None
    ):
        return None

    profile = _get_municipality_profile(loader, municipality_id)
    if profile is None:
        return None

    schema = loader.zoning_feature_schema
    feature_names = schema["feature_names"]

    # Start from the municipality profile values
    row: Dict[str, Any] = {}
    for col in feature_names:
        if col in profile.index:
            row[col] = profile[col]

    # Ensure cod_dane_m and cod_depart are present
    row["cod_dane_m"] = int(municipality_id)
    row["cod_depart"] = int(municipality_id[:2])

    # Crop and climate zone are required categoricals
    row["cultivo"] = crop_id
    row["zona_climatica"] = str(profile.get("zona_climatica", "templada"))

    # Department one-hot dummies
    dept_key = str(profile.get("departamento_key", "")).lower().replace(" ", "_")
    for col in feature_names:
        if col.startswith("depto_"):
            row[col] = 1 if col == f"depto_{dept_key}" else 0

    df = pd.DataFrame([row])

    # Encode categoricals
    df = _encode_zoning_categoricals(df, loader.zoning_preprocessor)

    # Reorder columns to match training
    missing = [c for c in feature_names if c not in df.columns]
    for col in missing:
        df[col] = 0
    df = df[feature_names]

    return df


def _get_yield_historical_records(
    loader: Any,
    municipality_id: str,
    crop_id: str,
) -> Optional[pd.DataFrame]:
    """Return historical EVA records for a municipality-crop pair.

    Falls back to department-crop averages and then national-crop averages when
    no municipality-level history exists, so the calendar endpoint can produce a
    signal for any supported crop.
    """
    profiles = loader.yield_profiles
    if profiles is None:
        return None
    try:
        dane_int = int(municipality_id)
    except ValueError:
        return None

    crop_name = _YIELD_CROP_NAME_MAP.get(crop_id)
    if not crop_name:
        return None

    dept_code = dane_int // 1000

    # 1. Municipality-crop history
    rows = profiles[
        (profiles["c_digo_dane_municipio"] == dane_int)
        & (profiles["cultivo"] == crop_name)
    ]
    if not rows.empty:
        return rows.sort_values(["año", "semestre"])

    # 2. Department-crop average (most recent synthetic record)
    dept_rows = profiles[
        (profiles["c_digo_dane_departamento"] == dept_code)
        & (profiles["cultivo"] == crop_name)
    ]
    if not dept_rows.empty:
        return _synthetic_profile(dept_rows, dane_int, dept_code)

    # 3. National-crop average
    national_rows = profiles[profiles["cultivo"] == crop_name]
    if not national_rows.empty:
        return _synthetic_profile(national_rows, dane_int, dept_code)

    return None


def _synthetic_profile(rows: pd.DataFrame, dane_int: int, dept_code: int) -> pd.DataFrame:
    """Build a single synthetic record from a group of historical records."""
    numeric = rows.select_dtypes(include=[np.number])
    means = numeric.mean().to_dict()
    means["c_digo_dane_municipio"] = dane_int
    means["c_digo_dane_departamento"] = dept_code

    # Use the most frequent categorical values
    for col in ["grupo_cultivo", "ciclo_del_cultivo", "estado_f_sico_del_cultivo", "clima_tipo"]:
        if col in rows.columns:
            means[col] = rows[col].mode().iloc[0] if not rows[col].mode().empty else ""

    # Pick a recent year/semester for ordering
    means["año"] = int(rows["año"].max()) if "año" in rows.columns else 2024
    means["semestre"] = int(rows["semestre"].max()) if "semestre" in rows.columns else 1

    return pd.DataFrame([means])


def _build_yield_historical_features(
    hist: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """Compute lag and moving-average features from historical records."""
    if hist is None or len(hist) < 1:
        return None

    rend = hist["rendimiento_winsorized"].values
    features: Dict[str, float] = {}

    # Current/most recent values
    last = hist.iloc[-1]
    features["rendimiento_lag1"] = float(rend[-1])
    features["rendimiento_historico_mean"] = float(np.mean(rend))
    features["rendimiento_historico_max"] = float(np.max(rend))
    features["rendimiento_historico_min"] = float(np.min(rend))

    # Lag 2 and 3 when available
    features["rendimiento_lag2"] = float(rend[-2]) if len(rend) >= 2 else features["rendimiento_lag1"]
    features["rendimiento_lag3"] = float(rend[-3]) if len(rend) >= 3 else features["rendimiento_lag2"]

    # Moving averages
    features["rendimiento_media_movil_2"] = float(np.mean(rend[-2:])) if len(rend) >= 2 else features["rendimiento_historico_mean"]
    features["rendimiento_media_movil_3"] = float(np.mean(rend[-3:])) if len(rend) >= 3 else features["rendimiento_historico_mean"]

    # Climate from the most recent record
    features["T2M"] = float(last.get("T2M", 0.0))
    features["PRECTOTCORR"] = float(last.get("PRECTOTCORR", 0.0))
    features["RH2M"] = float(last.get("RH2M", 0.0))
    features["ALLSKY_SFC_SW_DWN"] = float(last.get("ALLSKY_SFC_SW_DWN", 0.0))

    return features


def build_yield_features(
    db: Session,
    crop_id: str,
    municipality_id: str,
    loader: Any,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build feature matrices for the XGBoost and LightGBM yield models.

    Returns (X_xgb, X_lgbm) or None when historical records or artifacts are
    missing. Both matrices are aligned with their respective feature schemas.
    """
    if (
        not loader.is_yield_model_loaded()
        or loader.yield_preprocessor is None
        or loader.yield_feature_schema is None
        or loader.yield_profiles is None
    ):
        return None

    hist = _get_yield_historical_records(loader, municipality_id, crop_id)
    if hist is None:
        return None

    historical = _build_yield_historical_features(hist)
    if historical is None:
        return None

    schema = loader.yield_feature_schema
    xgb_schema = schema.get("xgboost", {})
    lgbm_schema = schema.get("lightgbm", {})

    xgb_features = xgb_schema.get("feature_names", [])
    lgbm_features = lgbm_schema.get("feature_names", [])

    if not xgb_features or not lgbm_features:
        return None

    last = hist.iloc[-1]
    dane_int = int(municipality_id)

    # Try to get lat/lon from municipality profiles; otherwise query the DB.
    lat, lon = None, None
    profile = _get_municipality_profile(loader, municipality_id)
    if profile is not None:
        lat = float(profile.get("latitud"))
        lon = float(profile.get("longitud"))
    else:
        from app.models import Municipality
        municipality = db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()
        if municipality is not None:
            lat = float(municipality.latitude) if municipality.latitude is not None else None
            lon = float(municipality.longitude) if municipality.longitude is not None else None

    lat = lat or 0.0
    lon = lon or 0.0

    # Common base values
    base: Dict[str, Any] = {
        "c_digo_dane_municipio": dane_int,
        "c_digo_dane_departamento": dane_int // 1000,
        "latitud": lat,
        "longitud": lon,
        "cultivo": _YIELD_CROP_NAME_MAP.get(crop_id, crop_id),
        "grupo_cultivo": str(last.get("grupo_cultivo", "")),
        "ciclo_del_cultivo": str(last.get("ciclo_del_cultivo", "Permanente")),
        "estado_f_sico_del_cultivo": str(last.get("estado_f_sico_del_cultivo", "En fresco")),
        "clima_tipo": str(last.get("clima_tipo", "anual")),
    }
    base.update(historical)

    # Add EcoCrop/FAO auxiliary features used by XGBoost
    crop_features = _get_crop_features_row(loader, crop_id)
    if crop_features is not None:
        aux_fields = [
            "alt_max",
            "temp_opt_max",
            "temp_opt_min",
            "rain_opt_min",
            "rain_opt_max",
            "ph_opt_min",
            "ph_opt_max",
            "gmin_dias",
            "gmax_dias",
            "cycle_duration_mean",
        ]
        for field in aux_fields:
            val = crop_features.get(field)
            if val is not None and not pd.isna(val):
                base[field] = float(val)

    # Model-specific extra fields
    # XGB uses aux features; LightGBM does not.
    base_xgb = base.copy()
    base_lgbm = base.copy()

    # Semester is only in LightGBM feature list
    base_lgbm["semestre"] = int(last.get("semestre", 0))

    # Sown area (we do not know current area; use last observed or 0)
    base_xgb["rea_sembrada_winsorized"] = float(last.get("rea_sembrada_winsorized", 0.0))
    base_lgbm["rea_sembrada_winsorized"] = float(last.get("rea_sembrada_winsorized", 0.0))

    # Year for model inputs (año) is not a feature, but derived ones are present

    df_xgb = pd.DataFrame([base_xgb])
    df_lgbm = pd.DataFrame([base_lgbm])

    # Encode categoricals with the fitted OrdinalEncoder
    cat_cols = ["grupo_cultivo", "cultivo", "ciclo_del_cultivo", "estado_f_sico_del_cultivo", "clima_tipo"]
    present_cat = [c for c in cat_cols if c in df_xgb.columns]
    if present_cat:
        df_xgb[present_cat] = loader.yield_preprocessor.transform(df_xgb[present_cat].astype(str))
        df_lgbm[present_cat] = loader.yield_preprocessor.transform(df_lgbm[present_cat].astype(str))

    # Reorder and fill missing columns
    for col in xgb_features:
        if col not in df_xgb.columns:
            df_xgb[col] = 0
    df_xgb = df_xgb[xgb_features]

    for col in lgbm_features:
        if col not in df_lgbm.columns:
            df_lgbm[col] = 0
    df_lgbm = df_lgbm[lgbm_features]

    return df_xgb, df_lgbm
