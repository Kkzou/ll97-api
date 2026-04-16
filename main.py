# -*- coding: utf-8 -*-
"""
main.py — LL97 Compliance Risk Prediction API
Deploy on Render: uvicorn main:app --host 0.0.0.0 --port 10000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI(
    title="LL97 Compliance Risk API",
    description="Neural network–based GHG risk classification for NYC LL97",
    version="1.0.0"
)

# ---------------------------------------------------
# CORS — allow your Vercel frontend
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ll97-compliance.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "*"  # remove this line after testing
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# LOAD ARTIFACTS AT STARTUP
# ---------------------------------------------------
A = Path("artifacts")

try:
    model         = joblib.load(A / "nn_model.pkl")
    scaler        = joblib.load(A / "scaler.pkl")
    imputer       = joblib.load(A / "imputer.pkl")
    label_encoder = joblib.load(A / "label_encoder.pkl")
    feat_cols     = joblib.load(A / "feature_columns.pkl")
    with open(A / "feature_means.json") as f:
        means = json.load(f)
    print("✓ All model artifacts loaded successfully")
except Exception as e:
    print(f"✗ Error loading artifacts: {e}")
    raise

# ---------------------------------------------------
# LL97 CAPS BY BUILDING TYPE (2024–2029)
# tCO2e per sqft per year
# ---------------------------------------------------
LL97_CAPS = {
    "Office":                        0.00846,
    "Education":                     0.00758,
    "Retail":                        0.01181,
    "Lodging/residential":           0.00675,
    "Healthcare":                    0.02381,
    "Entertainment/public assembly": 0.00765,
    "Warehouse/storage":             0.00403,
    "Manufacturing/industrial":      0.01181,
    "Public services":               0.00846,
    "Other":                         0.00846,
}

# ---------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------
class BuildingInput(BaseModel):
    sqft:                    Optional[float] = None
    primaryspaceusage:       Optional[str]   = "Office"
    yearbuilt:               Optional[float] = None
    airTemperature_winter:   Optional[float] = None
    airTemperature_summer:   Optional[float] = None
    dewTemperature_winter:   Optional[float] = None
    dewTemperature_summer:   Optional[float] = None

# ---------------------------------------------------
# HELPER
# ---------------------------------------------------
def fill(val, key):
    """Use provided value or fall back to dataset mean."""
    return val if val is not None else means.get(key, 0.0)

# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "LL97 Risk API is running"}

# ---------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------
@app.post("/predict")
def predict(b: BuildingInput):
    try:
        # Fill missing inputs with dataset means
        sqft      = fill(b.sqft,                  "sqft")
        yearbuilt = fill(b.yearbuilt,              "yearbuilt")
        air_w     = fill(b.airTemperature_winter,  "airTemperature_winter")
        air_s     = fill(b.airTemperature_summer,  "airTemperature_summer")
        dew_w     = fill(b.dewTemperature_winter,  "dewTemperature_winter")
        dew_s     = fill(b.dewTemperature_summer,  "dewTemperature_summer")
        usage     = b.primaryspaceusage or "Office"

        cap          = LL97_CAPS.get(usage, 0.00846)
        building_age = 2024 - yearbuilt

        # Pull remaining means for engineered features
        energy_intensity = means.get("energy_intensity_kWh_per_sqft", 39.9)
        ghg_intensity    = means.get("ghg_intensity", 0.006)
        total_energy     = energy_intensity * sqft
        occupants        = means.get("occupants", 263.0)
        floors           = max(means.get("numberoffloors", 3.6), 1)
        annual_cap       = cap * sqft
        cap_util         = ghg_intensity / cap if cap > 0 else 1.0
        cap_gap          = ghg_intensity - cap

        # Replicate full feature engineering from training
        row = {
            "sqft":                            sqft,
            "primaryspaceusage":               usage,
            "yearbuilt":                       yearbuilt,
            "occupants":                       occupants,
            "numberoffloors":                  floors,
            "energystarscore":                 means.get("energystarscore", 50.0),
            "energy_intensity_kWh_per_sqft":   energy_intensity,
            "ghg_intensity":                   ghg_intensity,
            "data_quality_weight_normalized":  means.get("data_quality_weight_normalized", 0.5),
            "building_age":                    building_age,
            "log_energy_intensity":            np.log1p(energy_intensity),
            "log_ghg_intensity":               np.log1p(ghg_intensity),
            "log_total_energy":                np.log1p(total_energy),
            "log_sqft":                        np.log1p(sqft),
            "electricity_ratio":               means.get("electricity_ratio", 0.5),
            "gas_ratio":                       means.get("gas_ratio", 0.3),
            "steam_ratio":                     means.get("steam_ratio", 0.2),
            "energy_per_floor":                energy_intensity * sqft / floors,
            "ghg_per_floor":                   ghg_intensity * sqft / floors,
            "energy_per_occupant":             total_energy / max(occupants, 1),
            "ghg_per_occupant":                ghg_intensity * sqft / max(occupants, 1),
            "occupants_per_sqft":              occupants / max(sqft, 1),
            "sqft_per_floor":                  sqft / floors,
            "energystar_inverse":              1 / max(means.get("energystarscore", 50.0), 1),
            "heating_severity_proxy":          max(0, 18 - air_w),
            "cooling_severity_proxy":          max(0, air_s - 24),
            "seasonal_temp_range":             air_s - air_w,
            "age_x_sqft":                      building_age * sqft,
            "age_x_energy_intensity":          building_age * energy_intensity,
            "occupancy_x_sqft":                occupants * sqft,
            "cap_utilization_ratio":           cap_util,
            "cap_gap":                         cap_gap,
            "annual_emissions_cap_tco2e":      annual_cap,
            "energy_missing_count":            means.get("energy_missing_count", 1.0),
            "building_age_bucket":             means.get("building_age_bucket", 2.0),
            "dominant_fuel":                   means.get("dominant_fuel", 0.0),
            "winter_temp_x_energy_intensity":  air_w * energy_intensity,
            "summer_temp_x_energy_intensity":  air_s * energy_intensity,
            "airTemperature_winter":           air_w,
            "airTemperature_summer":           air_s,
            "airTemperature_spring":           means.get("airTemperature_spring", 14.7),
            "airTemperature_fall":             means.get("airTemperature_fall", 16.7),
            "cloudCoverage_winter":            means.get("cloudCoverage_winter", 2.17),
            "cloudCoverage_summer":            means.get("cloudCoverage_summer", 2.23),
            "cloudCoverage_spring":            means.get("cloudCoverage_spring", 2.10),
            "cloudCoverage_fall":              means.get("cloudCoverage_fall", 1.79),
            "dewTemperature_winter":           dew_w,
            "dewTemperature_summer":           dew_s,
            "dewTemperature_spring":           means.get("dewTemperature_spring", 6.24),
            "dewTemperature_fall":             means.get("dewTemperature_fall", 10.03),
            "seaLvlPressure_winter":           means.get("seaLvlPressure_winter", 1018.0),
            "seaLvlPressure_summer":           means.get("seaLvlPressure_summer", 1014.3),
            "seaLvlPressure_spring":           means.get("seaLvlPressure_spring", 1015.6),
            "seaLvlPressure_fall":             means.get("seaLvlPressure_fall", 1017.1),
        }

        # Build dataframe and encode
        df_in = pd.DataFrame([row])
        df_in = pd.get_dummies(df_in, drop_first=True)

        # Align to exact training column schema
        for col in feat_cols:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[feat_cols]

        # Impute → scale → predict
        X = imputer.transform(df_in)
        X = scaler.transform(X)

        pred_int   = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        risk_label = label_encoder.inverse_transform([pred_int])[0]

        classes    = list(label_encoder.classes_)
        proba_dict = {c: round(float(p), 4) for c, p in zip(classes, pred_proba)}

        # Penalty calculation ($268/tCO2e above cap)
        penalty = 0.0
        if risk_label == "high":
            excess  = max(0.0, ghg_intensity - cap) * sqft
            penalty = round(excess * 268, 2)

        return {
            "risk":                          risk_label,
            "probabilities":                 proba_dict,
            "estimated_ghg_intensity":       round(float(ghg_intensity), 6),
            "ll97_cap":                      round(float(cap), 6),
            "cap_gap":                       round(float(cap_gap), 6),
            "estimated_annual_penalty_usd":  penalty,
            "inputs_used": {
                "sqft":      sqft,
                "yearbuilt": yearbuilt,
                "usage":     usage,
                "air_temp_winter": round(air_w, 2),
                "air_temp_summer": round(air_s, 2),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
