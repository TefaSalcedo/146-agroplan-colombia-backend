"""Model Loader for ML models from Hugging Face.

Downloads and caches model artifacts from HF repos with SHA-256 verification.
Models are loaded once during application lifespan and kept in memory.

When models are not configured or unavailable, the system falls back to
mock predictions with explicit method labeling.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import numpy as np

from app.config import get_settings
from app.logger import get_logger

# Make shared_eval (climate_analogs pickle dependency) importable during unpickling.
_ML_DIR = Path(__file__).resolve().parent.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

settings = get_settings()
logger = get_logger("app.services.model_loader")


# Ordinal mapping used by the zoning LightGBM model.
_ZONING_CLASS_ORDER = ["no_apta", "baja", "media", "alta"]


def _sha256_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_first_file(candidates: List[Path]) -> Optional[Path]:
    """Return the first existing file from a list of candidate paths."""
    for path in candidates:
        if path.exists():
            return path
    return None


def _missing_files(local_dir: Path, filenames: List[str]) -> List[str]:
    """Return the subset of filenames that do not exist in local_dir."""
    missing = []
    for name in filenames:
        if not (local_dir / name).exists():
            missing.append(name)
    return missing


def _download_hf_files(
    repo_id: str,
    local_dir: Path,
    filenames: List[str],
    revision: str = "main",
    token: str = "",
) -> bool:
    """Download a specific list of files from a Hugging Face model repo.

    Skips files that already exist locally so restarts are fast and network
    usage is minimal. Returns True if all requested files are present afterwards.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("[model_loader] huggingface_hub not installed; cannot download from HF")
        return False

    if not repo_id or not filenames:
        return False

    local_dir.mkdir(parents=True, exist_ok=True)

    # Only download what is actually missing
    to_download = _missing_files(local_dir, filenames)
    if not to_download:
        logger.info(f"[model_loader] All requested files already cached in {local_dir}")
        return True

    logger.info(f"[model_loader] Downloading {len(to_download)} missing file(s) from {repo_id}")
    try:
        for filename in to_download:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token or None,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        logger.info(f"[model_loader] Downloaded missing files from {repo_id} to {local_dir}")
        return True
    except Exception as e:
        logger.info(f"[model_loader] Failed to download files from {repo_id}: {e}")
        return False


def _resolve_yield_files(local_dir: Path) -> List[str]:
    """Return the minimal set of yield files needed for the MVP ensemble."""
    active_file = local_dir / "active_models.json"
    active_models = {}
    if active_file.exists():
        try:
            with open(active_file) as f:
                active_models = json.load(f)
        except Exception:
            pass

    files = ["active_models.json"]
    models = active_models or {
        "xgboost": {"path": "xgboost_top20_cleaned.pkl"},
        "lightgbm": {"path": "lightgbm_top20_cleaned.pkl"},
    }
    for key, meta in models.items():
        if key in ("xgboost", "lightgbm"):
            path = meta.get("path") if isinstance(meta, dict) else meta
            if path:
                files.append(path)
            log = meta.get("log") if isinstance(meta, dict) else None
            if log:
                files.append(log)

    # Bundle artifacts (uploaded separately by the data science workflow)
    files.extend([
        "yield/preprocessor.pkl",
        "yield/feature_schema.json",
        "yield/weights.json",
        "yield/manifest.json",
        # Profiles
        "models/yield_profiles.parquet",
        # Auxiliary crop/calendar datasets for feature building and calendars
        "data/crop_features_integrated.csv",
        "data/calendar_for_eva.csv",
    ])
    return files


def _resolve_zoning_files(local_dir: Path) -> List[str]:
    """Return the minimal set of zoning files needed for the MVP backend."""
    return [
        "lightgbm_tuned_multi_random_holdout.pkl",
        "catboost_tuned_multi_spatial_holdout.pkl",
        "climate_analog_recommender.pkl",
        # Bundle artifacts
        "zoning/preprocessor.pkl",
        "zoning/feature_schema.json",
        "zoning/catboost_feature_schema.json",
        "zoning/manifest.json",
        "zoning/golden_vectors.json",
        # Profiles
        "models/municipality_profiles.parquet",
        "models/zoning_reference.parquet",
    ]


class ModelLoader:
    """Loads and manages ML model artifacts from Hugging Face."""

    def __init__(self):
        self.zoning_model = None
        self.zoning_preprocessor = None
        self.zoning_feature_schema: Optional[Dict] = None
        self.climate_analog_recommender = None

        self.zoning_catboost_model = None
        self.zoning_catboost_preprocessor = None
        self.zoning_catboost_feature_schema: Optional[Dict] = None

        self.yield_xgb_model = None
        self.yield_lgbm_model = None
        self.yield_preprocessor = None
        self.yield_feature_schema: Optional[Dict] = None
        self.yield_xgb_weight: float = 0.65
        self.yield_lgbm_weight: float = 0.35

        self.knn_fallback = None
        self.municipality_profiles = None
        self.yield_profiles = None
        self.crop_features = None
        self.calendar_for_eva = None

        self._profiles_loaded = False
        self._golden_vectors_passed: Optional[bool] = None
        self._load_error: Optional[str] = None

        self._load_models()

    def _load_models(self):
        """Load models from local cache or download from HF.

        The default "mvp" mode downloads only the files required for the
        production endpoints, which keeps disk/RAM usage low on constrained
        hosts such as Oracle Cloud Free Tier.
        """
        models_dir = Path(settings.ml_models_path)
        models_dir.mkdir(parents=True, exist_ok=True)

        download_mode = getattr(settings, "hf_download_mode", "mvp").lower()
        logger.info("[model_loader] Starting _load_models (download_mode=%s)", download_mode)

        try:
            # Download from HF if repos are configured. This is best-effort:
            # failures are logged and the loader continues with local files/mock.
            if download_mode != "none" and settings.hf_model_repo_zoning:
                logger.info("[model_loader] Downloading zoning artifacts from HF repo=%s", settings.hf_model_repo_zoning)
                if download_mode == "all":
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=settings.hf_model_repo_zoning,
                        revision=settings.hf_model_revision,
                        token=settings.hf_token or None,
                        local_dir=str(models_dir / "zoning"),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                else:
                    zoning_files = _resolve_zoning_files(models_dir / "zoning")
                    _download_hf_files(
                        settings.hf_model_repo_zoning,
                        models_dir / "zoning",
                        filenames=zoning_files,
                        revision=settings.hf_model_revision,
                        token=settings.hf_token,
                    )

            if download_mode != "none" and settings.hf_model_repo_yield:
                if download_mode == "all":
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=settings.hf_model_repo_yield,
                        revision=settings.hf_model_revision,
                        token=settings.hf_token or None,
                        local_dir=str(models_dir / "yield"),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                else:
                    yield_files = _resolve_yield_files(models_dir / "yield")
                    _download_hf_files(
                        settings.hf_model_repo_yield,
                        models_dir / "yield",
                        filenames=yield_files,
                        revision=settings.hf_model_revision,
                        token=settings.hf_token,
                    )

            # Try to load zoning model
            logger.info("[model_loader] Loading zoning model into memory")
            self._load_zoning_model(models_dir)

            # Try to load yield models
            logger.info("[model_loader] Loading yield models into memory")
            self._load_yield_models(models_dir)

            # Try to load reference profiles
            logger.info("[model_loader] Loading reference profiles")
            self._load_profiles(models_dir)

            # Run golden vectors if models are loaded
            if self.is_zoning_model_loaded():
                logger.info("[model_loader] Running golden vector validation")
                self._run_golden_vectors()

        except Exception as e:
            self._load_error = str(e)
            logger.error("[model_loader] Error loading models: %s", e)

    def _load_zoning_model(self, models_dir: Path):
        """Load the zoning LightGBM and CatBoost models plus preprocessors."""
        zoning_path = models_dir / "zoning"
        bundle_path = zoning_path / "zoning"

        # --- LightGBM model (used by /zoning/recommendations) ---
        model_candidates = [
            zoning_path / "lightgbm_tuned_multi_random_holdout.pkl",
            zoning_path / "model.pkl",
            models_dir / "model.pkl",
        ]
        model_file = _find_first_file(model_candidates)
        preprocessor_file = _find_first_file([
            bundle_path / "preprocessor.pkl",
            zoning_path / "preprocessor.pkl",
        ])
        schema_file = _find_first_file([
            bundle_path / "feature_schema.json",
            zoning_path / "feature_schema.json",
        ])
        manifest_file = _find_first_file([
            bundle_path / "manifest.json",
            zoning_path / "manifest.json",
        ])

        if model_file:
            # Verify SHA-256 if manifest exists
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                expected_sha = manifest.get("sha256")
                if expected_sha:
                    actual_sha = _sha256_file(str(model_file))
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"Zoning model SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                        )

            self.zoning_model = joblib.load(str(model_file))
            logger.info(f"[model_loader] Loaded zoning model: {type(self.zoning_model).__name__}")
        else:
            logger.info("[model_loader] LightGBM zoning model not found")

        if preprocessor_file.exists():
            self.zoning_preprocessor = joblib.load(str(preprocessor_file))
            logger.info("[model_loader] Loaded zoning preprocessor")

        if schema_file.exists():
            with open(schema_file) as f:
                self.zoning_feature_schema = json.load(f)
            n_features = self.zoning_feature_schema.get("n_features", "unknown")
            logger.info(f"[model_loader] Loaded zoning feature schema: {n_features} features")

        # --- CatBoost model (used by /zoning/map) ---
        catboost_file = _find_first_file([
            zoning_path / "catboost_tuned_multi_spatial_holdout.pkl",
            models_dir / "catboost_tuned_multi_spatial_holdout.pkl",
        ])
        catboost_schema_file = _find_first_file([
            zoning_path / "zoning" / "catboost_feature_schema.json",
            zoning_path / "catboost_feature_schema.json",
            bundle_path / "catboost_feature_schema.json",
        ])

        if catboost_file:
            try:
                import pickle
                with open(catboost_file, "rb") as f:
                    self.zoning_catboost_model = pickle.load(f)
                logger.info(f"[model_loader] Loaded CatBoost zoning model: {type(self.zoning_catboost_model).__name__}")
            except Exception as e:
                logger.error(f"[model_loader] Could not load CatBoost zoning model: {e}")

        if catboost_schema_file and catboost_schema_file.exists():
            with open(catboost_schema_file) as f:
                self.zoning_catboost_feature_schema = json.load(f)
            n_features = self.zoning_catboost_feature_schema.get("n_features", "unknown")
            logger.info(f"[model_loader] Loaded CatBoost zoning feature schema: {n_features} features")

        # The same preprocessor encodes cultivo/zona_climatica for both models
        if self.zoning_catboost_model is not None and self.zoning_preprocessor is not None:
            self.zoning_catboost_preprocessor = self.zoning_preprocessor
            logger.info("[model_loader] Reusing zoning preprocessor for CatBoost")

        # Climate analog k-NN recommender (used as fallback for unseen municipalities)
        analog_file = _find_first_file([
            zoning_path / "climate_analog_recommender.pkl",
            models_dir / "climate_analog_recommender.pkl",
        ])
        if analog_file:
            try:
                logger.info("[model_loader] Loading climate analog recommender")
                import pickle
                with open(analog_file, "rb") as f:
                    self.climate_analog_recommender = pickle.load(f)
                logger.info(
                    f"[model_loader] Loaded climate analog recommender: "
                    f"{len(self.climate_analog_recommender.reference_df)} reference rows"
                )
            except Exception as e:
                logger.error("[model_loader] Could not load climate analog recommender: %s", e)

    def _load_yield_models(self, models_dir: Path):
        """Load the yield XGBoost and LightGBM models and ensemble weights."""
        yield_path = models_dir / "yield"
        bundle_path = yield_path / "yield"

        # Support both cleaned HF names and canonical bundle names
        xgb_candidates = [yield_path / "xgb_model.pkl", yield_path / "xgboost_top20_cleaned.pkl"]
        lgbm_candidates = [yield_path / "lgbm_model.pkl", yield_path / "lightgbm_top20_cleaned.pkl"]

        xgb_file = _find_first_file(xgb_candidates)
        lgbm_file = _find_first_file(lgbm_candidates)

        preprocessor_file = _find_first_file([
            bundle_path / "preprocessor.pkl",
            yield_path / "preprocessor.pkl",
        ])
        schema_file = _find_first_file([
            bundle_path / "feature_schema.json",
            yield_path / "feature_schema.json",
        ])
        weights_file = _find_first_file([
            bundle_path / "weights.json",
            yield_path / "weights.json",
        ])

        if xgb_file:
            self.yield_xgb_model = joblib.load(str(xgb_file))
            logger.info(f"[model_loader] Loaded yield XGBoost model: {type(self.yield_xgb_model).__name__}")

        if lgbm_file:
            self.yield_lgbm_model = joblib.load(str(lgbm_file))
            logger.info(f"[model_loader] Loaded yield LightGBM model: {type(self.yield_lgbm_model).__name__}")

        if preprocessor_file.exists():
            self.yield_preprocessor = joblib.load(str(preprocessor_file))
            logger.info("[model_loader] Loaded yield preprocessor")

        if schema_file.exists():
            with open(schema_file) as f:
                self.yield_feature_schema = json.load(f)
            xgb_n = len(self.yield_feature_schema.get("xgboost", {}).get("feature_names", []))
            lgbm_n = len(self.yield_feature_schema.get("lightgbm", {}).get("feature_names", []))
            logger.info(f"[model_loader] Loaded yield feature schema: XGB={xgb_n}, LGBM={lgbm_n} features")

        if weights_file.exists():
            with open(weights_file) as f:
                weights = json.load(f)
            self.yield_xgb_weight = float(
                weights.get("weight_xgb") or weights.get("xgboost") or 0.65
            )
            self.yield_lgbm_weight = float(
                weights.get("weight_lgb") or weights.get("lightgbm") or 0.35
            )
            logger.info(f"[model_loader] Loaded yield ensemble weights: XGB={self.yield_xgb_weight}, LGBM={self.yield_lgbm_weight}")
        else:
            self.yield_xgb_weight = 0.65
            self.yield_lgbm_weight = 0.35

    def _load_profiles(self, models_dir: Path):
        """Load reference Parquet profiles for k-NN fallback and feature building."""
        import pandas as pd

        zoning_models_dir = models_dir / "zoning" / "models"
        yield_models_dir = models_dir / "yield" / "models"
        yield_data_dir = models_dir / "yield" / "data"

        zoning_reference_file = _find_first_file([
            zoning_models_dir / "zoning_reference.parquet",
            models_dir / "models" / "zoning_reference.parquet",
        ])
        if zoning_reference_file:
            try:
                self.knn_fallback = pd.read_parquet(str(zoning_reference_file))
                logger.info(f"[model_loader] Loaded zoning reference profiles: {len(self.knn_fallback)} rows")
            except Exception as e:
                logger.info(f"[model_loader] Could not load zoning reference profiles: {e}")

        municipality_profiles_file = _find_first_file([
            zoning_models_dir / "municipality_profiles.parquet",
            models_dir / "models" / "municipality_profiles.parquet",
        ])
        if municipality_profiles_file:
            try:
                self.municipality_profiles = pd.read_parquet(str(municipality_profiles_file))
                logger.info(f"[model_loader] Loaded municipality profiles: {len(self.municipality_profiles)} rows")
            except Exception as e:
                logger.info(f"[model_loader] Could not load municipality profiles: {e}")

        yield_profiles_file = _find_first_file([
            yield_models_dir / "yield_profiles.parquet",
            models_dir / "models" / "yield_profiles.parquet",
        ])
        if yield_profiles_file:
            try:
                self.yield_profiles = pd.read_parquet(str(yield_profiles_file))
                logger.info(f"[model_loader] Loaded yield profiles: {len(self.yield_profiles)} rows")
            except Exception as e:
                logger.info(f"[model_loader] Could not load yield profiles: {e}")

        crop_features_file = _find_first_file([
            yield_data_dir / "crop_features_integrated.csv",
            models_dir / "data" / "crop_features_integrated.csv",
        ])
        if crop_features_file:
            try:
                self.crop_features = pd.read_csv(str(crop_features_file))
                logger.info(f"[model_loader] Loaded crop features: {len(self.crop_features)} rows")
            except Exception as e:
                logger.info(f"[model_loader] Could not load crop features: {e}")

        calendar_file = _find_first_file([
            yield_data_dir / "calendar_for_eva.csv",
            models_dir / "data" / "calendar_for_eva.csv",
        ])
        if calendar_file:
            try:
                self.calendar_for_eva = pd.read_csv(str(calendar_file))
                logger.info(f"[model_loader] Loaded EVA calendar: {len(self.calendar_for_eva)} rows")
            except Exception as e:
                logger.info(f"[model_loader] Could not load EVA calendar: {e}")

        if any(
            x is not None
            for x in [
                self.knn_fallback,
                self.municipality_profiles,
                self.yield_profiles,
                self.crop_features,
                self.calendar_for_eva,
            ]
        ):
            self._profiles_loaded = True

    def _run_golden_vectors(self):
        """Run golden vector validation to verify model integrity."""
        golden_file = _find_first_file([
            Path(settings.ml_models_path) / "zoning" / "zoning" / "golden_vectors.json",
            Path(settings.ml_models_path) / "zoning" / "golden_vectors.json",
        ])
        if not golden_file:
            logger.info("[model_loader] No golden vectors file found, skipping validation")
            return

        try:
            with open(golden_file) as f:
                golden = json.load(f)

            # Support both { "vectors": [...] } and a plain list of vectors
            vectors = golden.get("vectors") if isinstance(golden, dict) else golden
            if not vectors:
                self._golden_vectors_passed = True
                logger.info("[model_loader] No golden vectors to validate")
                return

            passed = 0
            failed = 0
            for vector in vectors:
                features = vector.get("features")
                expected_class = vector.get("expected_class")
                if features and expected_class is not None and self.zoning_model is not None:
                    import pandas as pd

                    df = pd.DataFrame([features])
                    schema_names = self.zoning_feature_schema.get("feature_names", [])
                    if schema_names:
                        df = df[[c for c in schema_names if c in df.columns]]
                    pred_idx = int(self.zoning_model.predict(df)[0])
                    pred_label = _ZONING_CLASS_ORDER[pred_idx]
                    if pred_label == expected_class:
                        passed += 1
                    else:
                        failed += 1

            self._golden_vectors_passed = failed == 0
            print(
                f"[model_loader] Golden vector validation: {passed} passed, {failed} failed"
            )
        except Exception as e:
            self._golden_vectors_passed = False
            logger.info(f"[model_loader] Golden vector validation failed: {e}")

    def is_zoning_model_loaded(self) -> bool:
        """Check if zoning model is loaded."""
        return self.zoning_model is not None

    def is_zoning_catboost_loaded(self) -> bool:
        """Check if the CatBoost zoning model and its artifacts are loaded."""
        return (
            self.zoning_catboost_model is not None
            and self.zoning_catboost_preprocessor is not None
            and self.zoning_catboost_feature_schema is not None
            and self.municipality_profiles is not None
        )

    def is_yield_model_loaded(self) -> bool:
        """Check if at least one yield model is loaded."""
        return self.yield_xgb_model is not None or self.yield_lgbm_model is not None

    def is_knn_fallback_available(self) -> bool:
        """Check if k-NN fallback profiles are loaded."""
        return self.knn_fallback is not None

    def is_climate_analog_loaded(self) -> bool:
        """Check if the climate analog k-NN recommender is loaded."""
        return self.climate_analog_recommender is not None

    def is_profiles_loaded(self) -> bool:
        return self._profiles_loaded

    def is_golden_vectors_passed(self) -> Optional[bool]:
        return self._golden_vectors_passed

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def predict_yield_ensemble(self, X: Any) -> Optional[float]:
        """Return ensemble prediction from XGBoost and LightGBM yield models.

        Expects X to be a DataFrame or array aligned with the trained features.
        Returns None if the models are not loaded.
        """
        if not self.is_yield_model_loaded():
            return None

        xgb_pred = None
        lgbm_pred = None

        if self.yield_xgb_model is not None:
            xgb_pred = float(self.yield_xgb_model.predict(X)[0])
        if self.yield_lgbm_model is not None:
            lgbm_pred = float(self.yield_lgbm_model.predict(X)[0])

        if xgb_pred is not None and lgbm_pred is not None:
            return float(
                self.yield_xgb_weight * xgb_pred + self.yield_lgbm_weight * lgbm_pred
            )
        if xgb_pred is not None:
            return xgb_pred
        if lgbm_pred is not None:
            return lgbm_pred
        return None

    def predict_zoning(self, X: Any) -> Optional[Any]:
        """Return class prediction and probabilities from the zoning LightGBM model.

        Expects X to be a DataFrame or array aligned with the trained features.
        Returns None if the model is not loaded.
        """
        if not self.is_zoning_model_loaded():
            return None
        return self.zoning_model.predict(X)

    def predict_zoning_proba(self, X: Any) -> Optional[np.ndarray]:
        """Return class probabilities from the zoning LightGBM model.

        Returns None if the model is not loaded.
        """
        if not self.is_zoning_model_loaded():
            logger.warning("[model_loader] Zoning model not loaded; cannot predict_proba")
            return None
        logger.debug("[model_loader] Running zoning model predict_proba")
        return self.zoning_model.predict_proba(X)

    def predict_zoning_catboost_map(
        self,
        crop_id: str,
    ) -> Optional[pd.DataFrame]:
        """Return a batch prediction for all municipalities for one crop.

        Returns a DataFrame with columns:
        - cod_dane_m
        - suitability
        - confidence
        - probabilities (dict)
        - method = 'catboost_batch'

        Returns None if the CatBoost model or municipality profiles are not loaded.
        """
        import pandas as pd

        if not self.is_zoning_catboost_loaded():
            logger.warning("[model_loader] CatBoost zoning model not loaded; cannot batch predict")
            return None

        logger.info("[model_loader] Running CatBoost batch prediction for crop=%s", crop_id)
        schema = self.zoning_catboost_feature_schema
        feature_names = schema["feature_names"]
        profiles = self.municipality_profiles.copy()

        # Add the crop column to every municipality row
        profiles["cultivo"] = crop_id

        # Make sure all schema columns exist
        for col in feature_names:
            if col not in profiles.columns:
                profiles[col] = 0

        df = profiles[feature_names].copy()

        # Encode categorical columns using the shared preprocessor
        categorical_features = schema.get("categorical_features", ["cultivo", "zona_climatica"])
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)

        encoded = self.zoning_catboost_preprocessor.transform(df[categorical_features])
        for i, col in enumerate(categorical_features):
            df[col] = encoded[:, i]

        X = df[feature_names]
        probabilities = self.zoning_catboost_model.predict_proba(X)

        class_order = schema.get("target_classes", ["no_apta", "baja", "media", "alta"])
        class_to_api = {"no_apta": "none", "baja": "low", "media": "medium", "alta": "high"}

        result_rows = []
        for idx, probas in enumerate(probabilities):
            class_idx = int(np.argmax(probas))
            class_label = class_order[class_idx]
            suitability = class_to_api.get(class_label, class_label)
            confidence = round(float(probas[class_idx]), 4)
            prob_dict = {
                class_to_api.get(class_order[i], class_order[i]): round(float(p), 4)
                for i, p in enumerate(probas)
            }
            result_rows.append({
                "cod_dane_m": int(profiles.iloc[idx]["cod_dane_m"]),
                "suitability": suitability,
                "confidence": confidence,
                "probabilities": prob_dict,
                "method": "catboost_batch",
            })

        return pd.DataFrame(result_rows)

    def get_status(self) -> Dict[str, Any]:
        """Return detailed status of all model components."""
        return {
            "zoning_model_loaded": self.is_zoning_model_loaded(),
            "zoning_preprocessor_loaded": self.zoning_preprocessor is not None,
            "zoning_schema_loaded": self.zoning_feature_schema is not None,
            "zoning_catboost_loaded": self.is_zoning_catboost_loaded(),
            "climate_analog_loaded": self.is_climate_analog_loaded(),
            "yield_xgb_loaded": self.yield_xgb_model is not None,
            "yield_lgbm_loaded": self.yield_lgbm_model is not None,
            "yield_preprocessor_loaded": self.yield_preprocessor is not None,
            "yield_schema_loaded": self.yield_feature_schema is not None,
            "knn_fallback_available": self.is_knn_fallback_available(),
            "municipality_profiles_loaded": self.municipality_profiles is not None,
            "yield_profiles_loaded": self.yield_profiles is not None,
            "crop_features_loaded": self.crop_features is not None,
            "calendar_for_eva_loaded": self.calendar_for_eva is not None,
            "profiles_loaded": self._profiles_loaded,
            "golden_vectors_passed": self._golden_vectors_passed,
            "load_error": self._load_error,
        }


# Singleton instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the singleton model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def reset_model_loader() -> None:
    """Reset the singleton (useful for testing)."""
    global _model_loader
    _model_loader = None
