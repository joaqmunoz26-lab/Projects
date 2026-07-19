from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DIR_EXP  = Path(__file__).resolve().parent
DIR_DATA = DIR_EXP / "data"
DIR_MOD  = DIR_EXP / "models"

sys.path.insert(0, str(DIR_EXP.parent.parent / "src"))
from importlib import import_module
FEATURES_MODELO = import_module("04_features").FEATURES_MODELO

def separar_x_y(df: pd.DataFrame, columna_objetivo: str = "descompensacion_glicemica_90d"):
    df = df.copy()
    if "sexo" in df.columns and df["sexo"].dtype == object:
        df["sexo"] = (df["sexo"].astype(str).str.upper() == "M").astype(int)
    y = df[columna_objetivo].astype(int)
    X = df[FEATURES_MODELO].copy()
    return X, y

def entrenar_xgboost(X_train, y_train, X_val, y_val) -> object:
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    print(f"  scale_pos_weight calculado: {spw:.4f}  "
          f"(produccion: 2.6881)")

    modelo = XGBClassifier(
        n_estimators=126,
        max_depth=4,
        learning_rate=0.011662890273931383,
        subsample=0.7301321323053057,
        colsample_bytree=0.7554709158757928,
        min_child_weight=3,
        reg_alpha=0.9384800715909529,
        reg_lambda=0.0013820379228636995,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        missing=np.nan,
        random_state=42,
        enable_categorical=False,
    )
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])
    print(f"  AUC-ROC val: {auc:.4f}")
    return modelo

def entrenar_forest(X_train, y_train, X_val, y_val) -> object:
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])
    print(f"  AUC-ROC val: {auc:.4f}")
    return modelo

def ejecutar():
    print("=" * 60)
    print("Experimento: Entrenamiento sobre split estratificado")
    print("=" * 60)

    train = pd.read_csv(DIR_DATA / "train_estratificado.csv")
    val   = pd.read_csv(DIR_DATA / "val_estratificado.csv")
    print(f"train: {len(train)} filas | val: {len(val)} filas")

    X_train, y_train = separar_x_y(train)
    X_val,   y_val   = separar_x_y(val)
    print(f"features: {X_train.shape[1]}")

    print("\n[1] XGBoost...")
    xgb = entrenar_xgboost(X_train, y_train, X_val, y_val)
    DIR_MOD.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"modelo": xgb, "columnas": list(X_train.columns)},
        DIR_MOD / "modelo_xgboost_estratificado.pkl",
    )
    print(f" Guardado: {DIR_MOD / 'modelo_xgboost_estratificado.pkl'}")

    print("\n[2] Random Forest...")
    rf = entrenar_forest(X_train, y_train, X_val, y_val)
    joblib.dump(
        {"modelo": rf, "columnas": list(X_train.columns)},
        DIR_MOD / "modelo_forest_estratificado.pkl",
    )
    print(f" Guardado: {DIR_MOD / 'modelo_forest_estratificado.pkl'}")
    print("\nEntrenamiento completado.")

if __name__ == "__main__":
    ejecutar()
