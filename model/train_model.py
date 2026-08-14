"""
Entraînement du modèle global de prédiction de la durée du prochain cycle menstruel.

Approche :
- Dataset : FedCycleData (cycles réels, 159 femmes, 1665 cycles)
- Pour chaque cycle (sauf le premier de chaque femme), on construit des features
  basées sur l'historique de la femme AVANT ce cycle (pas de fuite de données / data leakage)
- On compare une régression linéaire (baseline) à une régression Random Forest
- On sauvegarde le meilleur modèle avec joblib pour l'utiliser dans l'API
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rawFedCycleData.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "cycle_model.joblib")
METADATA_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit, pour chaque cycle, des features basées uniquement sur les cycles
    PRECEDENTS de la même femme (jamais le cycle actuel ni les cycles futurs).
    """
    df = df.sort_values(["ClientID", "CycleNumber"]).copy()

    grouped = df.groupby("ClientID")["LengthofCycle"]

    # Cycle juste avant
    df["cycle_precedent"] = grouped.shift(1)

    # Moyenne mobile des 3 derniers cycles (hors cycle actuel)
    df["moyenne_3_derniers"] = (
        grouped.shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # Écart-type des 3 derniers cycles (régularité de la femme)
    df["ecart_type_3_derniers"] = (
        grouped.shift(1).rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    )
    df["ecart_type_3_derniers"] = df["ecart_type_3_derniers"].fillna(0)

    # Numéro du cycle (peut capter une légère tendance dans le temps)
    df["cycle_number"] = df["CycleNumber"]

    # On garde uniquement les cycles où on a au moins un cycle précédent
    df = df.dropna(subset=["cycle_precedent"])

    return df


def main():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH)

    print("Construction des features (feature engineering)...")
    df_features = build_features(df)

    feature_cols = ["cycle_precedent", "moyenne_3_derniers", "ecart_type_3_derniers", "cycle_number"]
    target_col = "LengthofCycle"

    X = df_features[feature_cols]
    y = df_features[target_col]

    print(f"Nombre d'exemples d'entraînement disponibles : {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Baseline : régression linéaire ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, pred_lr)
    print(f"Régression linéaire — MAE : {mae_lr:.3f} jours")

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    print(f"Random Forest — MAE : {mae_rf:.3f} jours")

    # --- Baseline naïve pour comparaison : toujours prédire la moyenne globale ---
    mae_naive = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
    print(f"Baseline naïve (moyenne globale) — MAE : {mae_naive:.3f} jours")

    # On choisit le meilleur modèle entre les deux
    if mae_rf <= mae_lr:
        best_model = rf
        best_model_name = "RandomForestRegressor"
        best_mae = mae_rf
    else:
        best_model = lr
        best_model_name = "LinearRegression"
        best_mae = mae_lr

    print(f"\nModèle retenu : {best_model_name} (MAE = {best_mae:.3f} jours)")

    # Ré-entraînement final sur TOUTES les données disponibles (train+test)
    # pour maximiser l'usage des données une fois le choix du modèle validé
    best_model.fit(X, y)

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Modèle sauvegardé : {MODEL_OUTPUT_PATH}")

    metadata = {
        "model_name": best_model_name,
        "mae_days": round(best_mae, 3),
        "features": feature_cols,
        "target": target_col,
        "global_mean_cycle_length": round(float(y.mean()), 2),
        "global_std_cycle_length": round(float(y.std()), 2),
        "default_luteal_phase_days": 14,
        "training_examples": len(X),
    }
    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Métadonnées sauvegardées : {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
