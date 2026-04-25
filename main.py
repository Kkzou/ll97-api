# -*- coding: utf-8 -*-
"""
main.py — LL97 Compliance Risk Prediction API  v4.0

Two paths:
  - Energy bills provided  → direct LL97 calculation (no model)
  - Building features only → XGBoost classifier with threshold tuning

The classifier uses asymmetric decision thresholds optimized for
high recall on the 'high' risk class (catches 97% of high-risk buildings).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib, json
import numpy as np
import pandas as pd

app = FastAPI(title="LL97 Risk API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# LOAD MODEL B (XGBoost + threshold config)
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
    print("✓ Model B loaded successfully")
    print(f"  Method: {TCONF.get('best_method')}")
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
LL97_CAPS_2030 = {
    "Office": 0.00453, "Education": 0.00344, "Retail": 0.00403,
    "Lodging/residential": 0.00407, "Healthcare": 0.01193,
    "Entertainment/public assembly": 0.00426, "Warehouse/storage": 0.00110,
    "Manufacturing/industrial": 0.00574, "Public services": 0.00453, "Other": 0.00453,
}
ELEC_FACTOR  = 0.000288962
GAS_FACTOR   = 0.00005311
STEAM_FACTOR = 0.0000441
OIL_FACTOR   = 0.00007421
PENALTY_PER_TON = 268

# ---------------------------------------------------
# SCHEMA
# ---------------------------------------------------
class BuildingInput(BaseModel):
    sqft:                   float
    primaryspaceusage:      str
    yearbuilt:              Optional[float] = None
    numberoffloors:         Optional[float] = None
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
    return val if val is not None else MEANS.get(key, 0.0)

def predict_with_threshold(proba):
    """Apply asymmetric threshold tuning to bias toward 'high' detection."""
    high_idx = TCONF["high_idx"]
    low_idx  = TCONF["low_idx"]
    med_idx  = TCONF["medium_idx"]
    method   = TCONF.get("method", "argmax")

    p = proba[0]
    if method == "threshold_tuning" and TCONF.get("high_threshold"):
        t = TCONF["high_threshold"]
        if p[high_idx] >= t:
            return high_idx, p
        else:
            return (low_idx if p[low_idx] > p[med_idx] else med_idx), p
    else:
        # Fallback: pick argmax
        return int(np.argmax(p)), p

# ---------------------------------------------------
# ROOT
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "4.0.0",
        "model_ready": MODEL is not None,
        "model_method": TCONF.get("best_method") if MODEL else None,
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
        # Path 2 — No energy data → ML CLASSIFICATION
        # ====================================================
        if MODEL is None:
            raise HTTPException(503, "ML model not available")

        yearbuilt    = b.yearbuilt or 1970.0
        building_age = 2024 - yearbuilt
        log_sqft     = np.log1p(sqft)
        air_w        = fill(b.airTemperature_winter, "airTemperature_winter")
        air_s        = fill(b.airTemperature_summer, "airTemperature_summer")
        dew_w        = fill(b.dewTemperature_winter, "dewTemperature_winter")
        dew_s        = fill(b.dewTemperature_summer, "dewTemperature_summer")
        floors       = fill(b.numberoffloors, "numberoffloors")
        seasonal_gap = air_s - air_w

        row = {
            "primaryspaceusage":      usage,
            "sqft":                   sqft,
            "log_sqft":               log_sqft,
            "building_age":           building_age,
            "numberoffloors":         floors,
            "airTemperature_winter":  air_w,
            "airTemperature_summer":  air_s,
            "dewTemperature_winter":  dew_w,
            "dewTemperature_summer":  dew_s,
            "seasonal_temp_gap":      seasonal_gap,
        }

        df_in = pd.DataFrame([row])
        df_in = pd.get_dummies(df_in, drop_first=True)
        for col in FEATCOLS:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[FEATCOLS]

        X = IMPUTER.transform(df_in)

        proba = MODEL.predict_proba(X)
        pred_idx, p = predict_with_threshold(proba)
        risk = LE.inverse_transform([pred_idx])[0]

        classes = list(LE.classes_)
        proba_dict = {c: round(float(p[i]), 4) for i, c in enumerate(classes)}

        return {
            "risk": risk,
            "model_used": "Neural Network (XGBoost classifier)",
            "model_method": TCONF.get("best_method"),
            "probabilities": proba_dict,
            "ll97_cap": round(cap, 6),
            "estimated_ghg_intensity": None,
            "estimated_annual_emissions": None,
            "estimated_annual_penalty_usd": None,
            "note": "Classification only — to estimate exact emissions and penalties, provide your annual energy bills.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
