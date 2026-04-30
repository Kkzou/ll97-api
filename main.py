# -*- coding: utf-8 -*-
"""
main.py — LL97 Compliance Risk Prediction API  v5.0

Two paths:
  - Energy bills provided  → direct LL97 calculation (no model)
  - Building features only → Neural Network classifier (MLP 64-32-16)

The classifier returns soft probabilities for low / medium / high
based on building profile + climate features.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib, json
import numpy as np
import pandas as pd

app = FastAPI(title="LL97 Risk API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# LOAD NN MODEL + ARTIFACTS
# ---------------------------------------------------
A = Path("artifacts")
try:
    MODEL    = joblib.load(A / "nn_model.pkl")
    SCALER   = joblib.load(A / "scaler.pkl")
    IMPUTER  = joblib.load(A / "imputer.pkl")
    LE       = joblib.load(A / "label_encoder.pkl")
    FEATCOLS = joblib.load(A / "feature_columns.pkl")
    with open(A / "feature_means.json") as f:
        MEANS = json.load(f)
    with open(A / "threshold_config.json") as f:
        TCONF = json.load(f)
    print("✓ NN model loaded successfully")
    print(f"  Method: {TCONF.get('best_method')}")
    print(f"  Feature columns: {len(FEATCOLS)}")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    MODEL = None

# ---------------------------------------------------
# LL97 CONSTANTS
# ---------------------------------------------------
LL97_CAPS_2024 = {
    "Office": 0.00846, "Education": 0.00758, "Retail": 0.01181,
    "Lodging/residential": 0.00675, "Healthcare": 0.02381,
    "Entertainment/public assembly": 0.00765, "Warehouse/storage": 0.00403,
    "Manufacturing/industrial": 0.01181, "Public services": 0.00846, "Other": 0.00846,
}
ELEC_FACTOR  = 0.000288962
GAS_FACTOR   = 0.00005311
STEAM_FACTOR = 0.0000441
OIL_FACTOR   = 0.00007421
PENALTY_PER_TON = 268

# ---------------------------------------------------
# SCHEMA — accepts the 5 user-known fields, rest auto-filled
# ---------------------------------------------------
class BuildingInput(BaseModel):
    sqft:                   float
    primaryspaceusage:      str
    yearbuilt:              Optional[float] = None
    numberoffloors:         Optional[float] = None
    occupants:              Optional[float] = None
    energystarscore:        Optional[float] = None
    airTemperature_winter:  Optional[float] = None
    airTemperature_summer:  Optional[float] = None
    dewTemperature_winter:  Optional[float] = None
    dewTemperature_summer:  Optional[float] = None
    annual_electricity_kWh: Optional[float] = None
    annual_gas_kWh:         Optional[float] = None
    annual_steam_kWh:       Optional[float] = None
    annual_oil_kWh:         Optional[float] = None

# ---------------------------------------------------
# HELPER
# ---------------------------------------------------
def fill(val, key):
    """Use provided value or fall back to dataset mean."""
    return val if val is not None else MEANS.get(key, 0.0)

# ---------------------------------------------------
# ROOT
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "5.0.0",
        "model_ready": MODEL is not None,
        "model_type": TCONF.get("best_method") if MODEL else None,
        "n_features": len(FEATCOLS) if MODEL else None,
    }

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
@app.post("/predict")
def predict(b: BuildingInput):
    try:
        usage = b.primaryspaceusage or "Office"
        cap   = LL97_CAPS_2024.get(usage, 0.00846)
        sqft  = b.sqft

        # ====================================================
        # Path 1 — Energy bills provided → DIRECT CALCULATION
        # ====================================================
        has_energy = (
            b.annual_electricity_kWh is not None or
            b.annual_gas_kWh         is not None or
            b.annual_steam_kWh       is not None or
            b.annual_oil_kWh         is not None
        )
        if has_energy:
            elec  = b.annual_electricity_kWh or 0
            gas   = b.annual_gas_kWh         or 0
            steam = b.annual_steam_kWh       or 0
            oil   = b.annual_oil_kWh         or 0

            emissions = (elec*ELEC_FACTOR + gas*GAS_FACTOR
                       + steam*STEAM_FACTOR + oil*OIL_FACTOR)
            ghg = emissions / max(sqft, 1)
            ratio = ghg / cap

            if   ratio > 1.0:  risk = "high"
            elif ratio >= 0.8: risk = "medium"
            else:              risk = "low"

            penalty = (max(0, emissions - cap*sqft) * PENALTY_PER_TON
                       if risk == "high" else 0)

            return {
                "risk": risk,
                "model_used": "Direct calculation",
                "probabilities": {risk: 1.0},
                "ll97_cap": round(cap, 6),
                "estimated_ghg_intensity": round(ghg, 6),
                "estimated_annual_emissions": round(emissions, 2),
                "estimated_annual_penalty_usd": round(penalty, 2),
                "note": "Computed directly from your utility bills using LL97 emission factors.",
            }

        # ====================================================
        # Path 2 — No energy data → NN CLASSIFICATION
        # ====================================================
        if MODEL is None:
            raise HTTPException(503, "ML model not available")

        # User-known inputs (or dataset mean)
        yearbuilt    = b.yearbuilt or MEANS.get("yearbuilt", 1970.0)
        building_age = 2024 - yearbuilt
        floors       = fill(b.numberoffloors,        "numberoffloors")
        occupants    = fill(b.occupants,             "occupants")
        starscore    = fill(b.energystarscore,       "energystarscore")
        air_w        = fill(b.airTemperature_winter, "airTemperature_winter")
        air_s        = fill(b.airTemperature_summer, "airTemperature_summer")
        dew_w        = fill(b.dewTemperature_winter, "dewTemperature_winter")
        dew_s        = fill(b.dewTemperature_summer, "dewTemperature_summer")

        # Engineered features (recompute exactly as in training script)
        heating_severity = max(0, 18 - air_w)
        cooling_severity = max(0, air_s - 24)
        seasonal_range   = air_s - air_w
        age_x_sqft       = building_age * sqft
        occupancy_x_sqft = occupants    * sqft

        # Building age bucket: matches training (numeric bucket)
        if   building_age <  20: age_bucket = 0
        elif building_age <  40: age_bucket = 1
        elif building_age <  60: age_bucket = 2
        elif building_age <  80: age_bucket = 3
        else:                    age_bucket = 4

        # Build the row in EXACT training feature order
        row = {
            "sqft":                            sqft,
            "primaryspaceusage":               usage,
            "yearbuilt":                       yearbuilt,
            "building_age":                    building_age,
            "numberoffloors":                  floors,
            "occupants":                       occupants,
            "energystarscore":                 starscore,
            "data_quality_weight_normalized":  MEANS.get("data_quality_weight_normalized", 0.5),
            "building_age_bucket":             age_bucket,
            "airTemperature_winter":           air_w,
            "airTemperature_summer":           air_s,
            "airTemperature_spring":           MEANS.get("airTemperature_spring", 14.7),
            "airTemperature_fall":             MEANS.get("airTemperature_fall", 16.7),
            "cloudCoverage_winter":            MEANS.get("cloudCoverage_winter", 2.17),
            "cloudCoverage_summer":            MEANS.get("cloudCoverage_summer", 2.23),
            "cloudCoverage_spring":            MEANS.get("cloudCoverage_spring", 2.10),
            "cloudCoverage_fall":              MEANS.get("cloudCoverage_fall",   1.79),
            "dewTemperature_winter":           dew_w,
            "dewTemperature_summer":           dew_s,
            "dewTemperature_spring":           MEANS.get("dewTemperature_spring", 6.24),
            "dewTemperature_fall":             MEANS.get("dewTemperature_fall",  10.03),
            "seaLvlPressure_winter":           MEANS.get("seaLvlPressure_winter", 1018.0),
            "seaLvlPressure_summer":           MEANS.get("seaLvlPressure_summer", 1014.3),
            "seaLvlPressure_spring":           MEANS.get("seaLvlPressure_spring", 1015.6),
            "seaLvlPressure_fall":             MEANS.get("seaLvlPressure_fall",   1017.1),
            "heating_severity_proxy":          heating_severity,
            "cooling_severity_proxy":          cooling_severity,
            "seasonal_temp_range":             seasonal_range,
            "age_x_sqft":                      age_x_sqft,
            "occupancy_x_sqft":                occupancy_x_sqft,
        }

        df_in = pd.DataFrame([row])
        df_in = pd.get_dummies(df_in, drop_first=True)

        # Align to exact training feature schema
        for col in FEATCOLS:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[FEATCOLS]

        # NN pipeline: impute → scale → predict
        X = IMPUTER.transform(df_in)
        X = SCALER.transform(X)

        proba = MODEL.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        risk = LE.inverse_transform([pred_idx])[0]

        classes = list(LE.classes_)
        proba_dict = {c: round(float(proba[i]), 4) for i, c in enumerate(classes)}

        return {
            "risk": risk,
            "model_used": "Neural Network (MLP 64-32-16)",
            "model_method": TCONF.get("best_method"),
            "probabilities": proba_dict,
            "ll97_cap": round(cap, 6),
            "estimated_ghg_intensity": None,
            "estimated_annual_emissions": None,
            "estimated_annual_penalty_usd": None,
            "note": "Classification only. For exact emissions and penalty estimates, provide your annual energy bills.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
