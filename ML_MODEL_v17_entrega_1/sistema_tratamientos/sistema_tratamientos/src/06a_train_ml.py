
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")

DIR_BASE = Path(__file__).resolve().parent.parent
DIR_MODELOS = DIR_BASE / "models"
DIR_MLRUNS = DIR_BASE / "mlruns"

MODELOS_DISPONIBLES = [
    "logistic", "elasticnet", "tree", "forest",
    "xgboost", "lightgbm",
]

def entrenar_logistic(X_train, y_train, X_val, y_val):
    print("\n[1/7] Logistic Regression (L2)...")
    with mlflow.start_run(run_name="logistic_regression"):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)

        modelo = LogisticRegression(
            max_iter=1000, C=0.1, penalty="l2",
            class_weight="balanced", random_state=42,
        )
        modelo.fit(X_tr, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_va)[:, 1])

        mlflow.log_params({"modelo": "LogisticRegression", "C": 0.1, "penalty": "l2"})
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": modelo, "scaler": scaler, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_logistic.pkl",
        )
        print(f"  AUC val: {auc:.4f}")
        return auc

def entrenar_elasticnet(X_train, y_train, X_val, y_val):
    print("\n[2/7] Elastic Net Logistic (L1+L2)...")
    with mlflow.start_run(run_name="elastic_net_logistic"):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)

        modelo = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            l1_ratio=0.5, alpha=1e-4, max_iter=1000,
            class_weight="balanced", random_state=42,
        )
        modelo.fit(X_tr, y_train)

        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(modelo))
        cal.fit(X_tr, y_train)
        auc = roc_auc_score(y_val, cal.predict_proba(X_va)[:, 1])

        mlflow.log_params({"modelo": "ElasticNetLogistic", "l1_ratio": 0.5, "alpha": 1e-4})
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": cal, "scaler": scaler, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_elasticnet.pkl",
        )
        print(f"  AUC val: {auc:.4f}")
        return auc

def entrenar_tree(X_train, y_train, X_val, y_val):
    print("\n[3/7] Decision Tree...")
    with mlflow.start_run(run_name="decision_tree"):
        modelo = DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        )
        modelo.fit(X_train, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

        mlflow.log_params({"modelo": "DecisionTree", "max_depth": 8})
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": modelo, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_tree.pkl",
        )
        print(f"  AUC val: {auc:.4f}")
        return auc

def entrenar_forest(X_train, y_train, X_val, y_val):
    print("\n[4/7] Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        modelo = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        modelo.fit(X_train, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

        mlflow.log_params({"modelo": "RandomForest", "n_estimators": 300, "max_depth": 10})
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": modelo, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_forest.pkl",
        )
        print(f"  AUC val: {auc:.4f}")
        return auc

def _objetivo_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "missing": np.nan,
        "random_state": 42, "eval_metric": "auc",
    }
    modelo = XGBClassifier(**params)
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

def entrenar_xgboost(X_train, y_train, X_val, y_val, n_trials=30):
    print(f"\n[5/7] XGBoost con Optuna ({n_trials} trials)...")
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    estudio.optimize(
        lambda t: _objetivo_xgboost(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials, show_progress_bar=False,
    )

    with mlflow.start_run(run_name="xgboost_optimizado"):
        scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        modelo = XGBClassifier(
            **estudio.best_params,
            missing=np.nan,
            scale_pos_weight=scale_pw,
            random_state=42, eval_metric="auc",
        )
        modelo.fit(X_train, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

        for k, v in estudio.best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": modelo, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_xgboost.pkl",
        )
        print(f"  AUC val: {auc:.4f}  (mejor Optuna: {estudio.best_value:.4f})")
        return auc

def _objetivo_lightgbm(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42, "verbose": -1,
    }
    modelo = LGBMClassifier(**params)
    modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

def entrenar_lightgbm(X_train, y_train, X_val, y_val, n_trials=30):
    print(f"\n[6/7] LightGBM con Optuna ({n_trials} trials)...")
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    estudio.optimize(
        lambda t: _objetivo_lightgbm(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials, show_progress_bar=False,
    )

    with mlflow.start_run(run_name="lightgbm_optimizado"):
        modelo = LGBMClassifier(
            **estudio.best_params, class_weight="balanced",
            random_state=42, verbose=-1,
        )
        modelo.fit(X_train, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

        for k, v in estudio.best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("auc_val", auc)

        joblib.dump(
            {"modelo": modelo, "columnas": list(X_train.columns)},
            DIR_MODELOS / "modelo_lightgbm.pkl",
        )
        print(f"  AUC val: {auc:.4f}  (mejor Optuna: {estudio.best_value:.4f})")
        return auc

def ejecutar_entrenamiento(n_trials: int = 30, modelos: list = None):
    print("=" * 60)
    print("Entrenamiento bateria de modelos")
    print("=" * 60)

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(DIR_MLRUNS.absolute().as_uri())
    mlflow.set_experiment("descompensacion_glicemica_90d")

    df_train = pd.read_csv(DIR_BASE / "data/processed/train.csv")
    df_val = pd.read_csv(DIR_BASE / "data/processed/val.csv")

    X_train, y_train = _split.separar_x_y(df_train)
    X_val, y_val = _split.separar_x_y(df_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    print(f"Train: {len(df_train)} filas, val: {len(df_val)} filas")
    print(f"Features: {X_train.shape[1]}")
    print(f"Tasa evento train: {y_train.mean():.2%} | val: {y_val.mean():.2%}")

    if modelos is None:
        modelos = MODELOS_DISPONIBLES

    resultados = {}

    if "logistic" in modelos:
        resultados["Logistic Regression"] = entrenar_logistic(X_train, y_train, X_val, y_val)
    if "elasticnet" in modelos:
        resultados["Elastic Net"] = entrenar_elasticnet(X_train, y_train, X_val, y_val)
    if "tree" in modelos:
        resultados["Decision Tree"] = entrenar_tree(X_train, y_train, X_val, y_val)
    if "forest" in modelos:
        resultados["Random Forest"] = entrenar_forest(X_train, y_train, X_val, y_val)
    if "xgboost" in modelos:
        resultados["XGBoost"] = entrenar_xgboost(X_train, y_train, X_val, y_val, n_trials)
    if "lightgbm" in modelos:
        resultados["LightGBM"] = entrenar_lightgbm(X_train, y_train, X_val, y_val, n_trials)

    print("\n" + "=" * 60)
    print("Resumen del entrenamiento (AUC val)")
    print("=" * 60)
    for nombre, auc in sorted(resultados.items(), key=lambda x: -x[1]):
        print(f"  {nombre:25s}: {auc:.4f}")

    resumen = pd.DataFrame([
        {"modelo": n, "auc_val": a} for n, a in resultados.items()
    ]).sort_values("auc_val", ascending=False)
    resumen.to_csv(DIR_MODELOS / "resumen_entrenamiento.csv", index=False)

    print(f"\nModelos guardados en: {DIR_MODELOS}")
    print("Ejecutá luego: python src/06c_compare_models.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--modelos", nargs="+", default=None,
                        choices=MODELOS_DISPONIBLES,
                        help="Subconjunto de modelos a entrenar")
    args = parser.parse_args()
    ejecutar_entrenamiento(args.n_trials, args.modelos)
