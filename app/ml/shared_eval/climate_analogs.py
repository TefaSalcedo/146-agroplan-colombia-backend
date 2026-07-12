"""
Climate analogs for Fase 5: Modelo 1 Zonificacion.

Recommends crops for a (departamento, municipio, current-or-future climate)
query by finding k nearest neighbors in the climate+soil feature space among
records where crops are known to be apta (alta/media), and aggregating which
crops prosper under similar conditions.

This is the change-climate component: it works for regions with no historical
zonification for a given crop, by analogy to regions with similar climate+soil
where that crop is known to be apta. Complements the classifier (which can only
predict aptitude for crops it has seen).
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .eval_config import (
    ANALOGS_DIR,
    ANALOG_FEATURES,
    ANALOG_K_NEIGHBORS,
    CLASS_TO_INT,
    INT_TO_CLASS,
)


@dataclass
class ClimateAnalogRecommender:
    """k-NN recommender over climate+soil features.

    Built from the subset of records where aptitude is alta or media (crops
    known to prosper), so neighbors point to "places where things grow well".
    """
    scaler: StandardScaler
    nn: NearestNeighbors
    reference_df: pd.DataFrame  # the rows the NN was fit on (post-scaling index)
    feature_names: List[str]
    k: int = ANALOG_K_NEIGHBORS

    def recommend(
        self,
        query: pd.DataFrame,
        k: Optional[int] = None,
        min_apta_classes: Tuple[str, ...] = ('alta', 'media'),
    ) -> List[Dict]:
        """For each query row, return the recommended crops with scores.

        Score = share of k neighbors (in alta/media) that have that crop,
        weighted optionally by the aptitude class of the neighbor
        (alta=1.0, media=0.5).

        Returns a list (one entry per query row) of dicts:
            {'top_crops': [(crop, score), ...], 'neighbors': reference_df.iloc[idx]}
        """
        k = k or self.k
        Xq = self.scaler.transform(query[self.feature_names].fillna(0))
        distances, indices = self.nn.kneighbors(Xq, n_neighbors=min(k, len(self.reference_df)))

        class_weight = {'alta': 1.0, 'media': 0.5, 'baja': 0.2, 'no_apta': 0.0}
        results = []
        for i in range(len(query)):
            nbr_rows = self.reference_df.iloc[indices[i]]
            # Only count neighbors that are apta (alta/media) for scoring
            mask = nbr_rows['clase_predominante'].map(class_weight).fillna(0) > 0
            nbr_apta = nbr_rows[mask]
            if len(nbr_apta) == 0:
                results.append({'top_crops': [], 'n_neighbors': len(nbr_rows), 'n_apta': 0})
                continue
            # Weighted vote per crop
            weights = nbr_apta['clase_predominante'].map(class_weight).fillna(0).values
            crop_scores = (
                nbr_apta.assign(_w=weights)
                .groupby('cultivo')['_w'].sum()
                .sort_values(ascending=False)
            )
            total = crop_scores.sum()
            top = [(c, float(s / total)) for c, s in crop_scores.items()]
            results.append({
                'top_crops': top,
                'n_neighbors': len(nbr_rows),
                'n_apta': int(len(nbr_apta)),
            })
        return results

    def recommend_for_region(
        self,
        climate_soil_row: pd.DataFrame,
        k: Optional[int] = None,
    ) -> Dict:
        """Convenience: recommend crops for a single region row."""
        res = self.recommend(climate_soil_row, k=k)
        return res[0] if res else {'top_crops': []}


def build_analog_recommender(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    k: int = ANALOG_K_NEIGHBORS,
    apta_classes: Tuple[str, ...] = ('alta', 'media'),
    output_dir: Optional[Path] = None,
) -> ClimateAnalogRecommender:
    """Build (and save) a climate+soil k-NN recommender.

    df must contain: the analog features, 'cultivo', and 'clase_predominante'
    (as the integer-encoded target OR the string class). We map back to string.
    """
    if feature_names is None:
        feature_names = [c for c in ANALOG_FEATURES if c in df.columns]
    if output_dir is None:
        output_dir = ANALOGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize target to string class for filtering
    tgt = df['clase_predominante']
    if pd.api.types.is_integer_dtype(tgt):
        tgt_str = tgt.map(INT_TO_CLASS)
    else:
        tgt_str = tgt.astype(str).str.strip().str.lower()
    df = df.copy()
    df['clase_predominante'] = tgt_str

    # Keep only apta records as the reference set (places where crops prosper)
    ref = df[df['clase_predominante'].isin(apta_classes)].reset_index(drop=True)
    print(f"Climate analogs: {len(ref)} reference rows (apta={apta_classes}) "
          f"from {len(df)} total, features={len(feature_names)}")

    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref[feature_names].fillna(0))
    nn = NearestNeighbors(n_neighbors=min(k, len(ref)), metric='euclidean', n_jobs=-1)
    nn.fit(X_ref)

    recommender = ClimateAnalogRecommender(
        scaler=scaler, nn=nn, reference_df=ref, feature_names=feature_names, k=k,
    )

    with open(output_dir / 'climate_analog_recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    log = {
        'n_reference': len(ref),
        'n_total': len(df),
        'apta_classes': list(apta_classes),
        'features': feature_names,
        'k': k,
        'crops_in_reference': sorted(ref['cultivo'].unique().tolist()),
    }
    with open(output_dir / 'climate_analog_log.json', 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Saved climate analog recommender: {output_dir / 'climate_analog_recommender.pkl'}")
    return recommender


def load_analog_recommender(path: Optional[Path] = None) -> ClimateAnalogRecommender:
    if path is None:
        path = ANALOGS_DIR / 'climate_analog_recommender.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)
