"""Shared feature engineering used by every model script from step 5 onward.

All features are computed row-wise from the raw sensor columns, so there is
no leakage between train and test. They are computed BEFORE imputation, so a
missing input simply gives a missing feature (the pipeline imputer handles it).

Features dropped after the step-5 analysis (single-feature AUC ~0.5-0.6, no CV
gain): power_gap, n_missing, temp_ratio, torque_per_speed, calc_power.
"""
import numpy as np
import pandas as pd

RAW_NUM = ["ambient_temp", "process_temp", "coolant_flow", "shaft_speed",
           "drive_torque", "power_load", "component_wear", "thermal_stress",
           "mechanical_load", "vibration_index"]
CAT = ["unit_type"]

EPS = 1e-6


def _div(a, b):
    return a / (b.abs() + EPS)


FEATURES = {
    # thermal side
    "temp_diff":      lambda d: d["process_temp"] - d["ambient_temp"],
    "power_per_cool": lambda d: _div(d["power_load"], d["coolant_flow"]),
    "ts_per_cool":    lambda d: _div(d["thermal_stress"], d["coolant_flow"]),
    "ts_per_tdiff":   lambda d: _div(d["thermal_stress"], d["process_temp"] - d["ambient_temp"]),
    "tdiff_per_cool": lambda d: _div(d["process_temp"] - d["ambient_temp"], d["coolant_flow"]),
    # mechanical side
    "wear_x_vib":     lambda d: d["component_wear"] * d["vibration_index"],
    "wear_x_load":    lambda d: d["component_wear"] * d["mechanical_load"],
    "vib_per_speed":  lambda d: _div(d["vibration_index"], d["shaft_speed"]),
    "ml_per_torque":  lambda d: _div(d["mechanical_load"], d["drive_torque"]),
}

# Named feature sets (used for the step-5 ablation and by later steps)
FEATURE_SETS = {
    "raw": [],
    "thermal3": ["ts_per_cool", "ts_per_tdiff", "tdiff_per_cool"],
    "thermal5": ["temp_diff", "power_per_cool", "ts_per_cool", "ts_per_tdiff", "tdiff_per_cool"],
    "mech4": ["wear_x_vib", "wear_x_load", "vib_per_speed", "ml_per_torque"],
    "all9": list(FEATURES),
}


def add_features(df, names):
    """Return a copy of df with the named engineered features added."""
    out = df.copy()
    for n in names:
        out[n] = FEATURES[n](df)
    return out
