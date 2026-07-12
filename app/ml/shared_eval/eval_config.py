"""Minimal shared_eval config required to unpickle ClimateAnalogRecommender.

This module is intentionally small: it only defines the constants referenced by
the pickled ``ClimateAnalogRecommender`` objects produced by the Fase 5
evaluation pipeline. The original eval_config.py resolves project paths; here we
keep only the constants needed at inference time.
"""

from pathlib import Path

# Dummy path kept for unpickle compatibility only.
ANALOGS_DIR = Path(".")

# Climate analogs configuration (k-NN over clima + suelo)
ANALOG_FEATURES = [
    't2m', 'prectotcorr', 'rh2m', 'allsky_sfc_sw_dwn',
    'ph_agua_suelo_mean', 'materia_organica_mean', 'fosforo_bray_ii_mean',
    'calcio_intercambiable_mean', 'magnesio_intercambiable_mean',
    'potasio_intercambiable_mean', 'capacidad_intercambio_cationico_mean',
    'conductividad_electrica_mean',
]
ANALOG_K_NEIGHBORS = 5
ANALOG_DISTANCE_METRIC = 'euclidean'

# Class mapping constants kept for compatibility.
CLASS_ORDER = ['no_apta', 'baja', 'media', 'alta']
INT_TO_CLASS = {0: 'no_apta', 1: 'baja', 2: 'media', 3: 'alta'}
CLASS_TO_INT = {v: k for k, v in INT_TO_CLASS.items()}
