"""
ml_predictor.py
"Use machine learning to predict optimal etching parameters for
desired profile accuracy" (Key Objectives) + "Basic AI Prediction"
(Key Features) + AI Models layer (Regression / etch-rate prediction)
from the System Architecture slide.

Two models are trained on physics-simulated + noisy data (standing in
for real sensor/process logs, per the Data Collection methodology
phase):
  1. LinearRegression  - simple, interpretable baseline (matches the
     "basic AI prediction" feature).
  2. RandomForestRegressor - captures the model's nonlinear
     power/pressure/flow interactions for better accuracy.

`recommend_parameters()` then does a constrained grid search over the
trained surrogate to find process conditions hitting a target etch
depth (the "optimal parameters" objective) without needing a new
physical experiment for every candidate point.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from config import MATERIALS, PROCESS_BOUNDS
from calculate_etch_depth import etch_rate


def make_training_data(material="Si", n_samples=1500, noise_std=0.05, seed=42):
    """
    Simulate a labeled dataset: random process conditions -> physics-model
    etch rate + measurement noise (stands in for logged fab/sensor data).
    """
    rng = np.random.default_rng(seed)
    rf = rng.uniform(*PROCESS_BOUNDS["rf_power"], n_samples)
    p = rng.uniform(*PROCESS_BOUNDS["pressure"], n_samples)
    q = rng.uniform(*PROCESS_BOUNDS["gas_flow"], n_samples)

    true_rate = np.array([etch_rate(material, r, pp, qq) for r, pp, qq in zip(rf, p, q)])
    noisy_rate = true_rate * (1 + rng.normal(0, noise_std, n_samples))

    df = pd.DataFrame({
        "rf_power": rf, "pressure": p, "gas_flow": q, "etch_rate_nm_min": noisy_rate
    })
    return df


def train_models(material="Si", n_samples=1500, test_size=0.2, seed=42):
    df = make_training_data(material, n_samples, seed=seed)
    X = df[["rf_power", "pressure", "gas_flow"]].values
    y = df["etch_rate_nm_min"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    lin = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=seed
    ).fit(X_train, y_train)

    results = {}
    for name, model in [("LinearRegression", lin), ("RandomForest", rf_model)]:
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, pred),
            "r2": r2_score(y_test, pred),
        }
    return results, (X_test, y_test)


def recommend_parameters(material, target_depth_nm, target_time_min, model,
                          n_grid=25):
    """
    Grid-search the trained surrogate model over the allowed process
    window to find (rf_power, pressure, gas_flow) whose predicted etch
    rate * target_time_min lands closest to target_depth_nm.
    """
    target_rate = target_depth_nm / target_time_min
    rf_vals = np.linspace(*PROCESS_BOUNDS["rf_power"], n_grid)
    p_vals = np.linspace(*PROCESS_BOUNDS["pressure"], n_grid)
    q_vals = np.linspace(*PROCESS_BOUNDS["gas_flow"], n_grid)

    grid = np.array(np.meshgrid(rf_vals, p_vals, q_vals)).reshape(3, -1).T
    preds = model.predict(grid)
    best_idx = np.argmin(np.abs(preds - target_rate))
    best = grid[best_idx]
    return {
        "rf_power": best[0],
        "pressure": best[1],
        "gas_flow": best[2],
        "predicted_rate_nm_min": preds[best_idx],
        "predicted_depth_nm": preds[best_idx] * target_time_min,
    }


if __name__ == "__main__":
    results, (X_test, y_test) = train_models("Si")
    for name, r in results.items():
        print(f"{name}: MAE={r['mae']:.2f} nm/min, R2={r['r2']:.4f}")

    best_model = results["RandomForest"]["model"]
    rec = recommend_parameters("Si", target_depth_nm=500, target_time_min=3, model=best_model)
    print("\nRecommended parameters for 500nm depth in 3 min:")
    for k, v in rec.items():
        print(f"  {k}: {v:.2f}")
