from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
)

DIR_EXP     = Path(__file__).resolve().parent
DIR_DATA    = DIR_EXP / "data"
DIR_MOD     = DIR_EXP / "models"
DIR_RESULTS = DIR_EXP / "results"
DIR_BASE    = DIR_EXP.parent.parent
RUTA_PROD   = DIR_BASE / "reports" / "tabla_comparativa_oficial.csv"

import sys
sys.path.insert(0, str(DIR_BASE / "src"))
from importlib import import_module
FEATURES_MODELO = import_module("04_features").FEATURES_MODELO

def separar_x_y(df: pd.DataFrame, columna_objetivo: str = "descompensacion_glicemica_90d"):
    df = df.copy()
    if "sexo" in df.columns and df["sexo"].dtype == object:
        df["sexo"] = (df["sexo"].astype(str).str.upper() == "M").astype(int)
    y = df[columna_objetivo].astype(int)
    X = df[FEATURES_MODELO].copy()
    return X, y

def calcular_metricas(nombre: str, y_true, y_prob) -> dict:
    umbral = 0.5
    y_pred = (y_prob >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / max(tp + fn, 1)
    esp  = tn / max(tn + fp, 1)

    return {
        "modelo":         nombre,
        "auc_roc":        round(roc_auc_score(y_true, y_prob),          6),
        "auc_pr":         round(average_precision_score(y_true, y_prob), 6),
        "sensibilidad":   round(sens, 6),
        "especificidad":  round(esp, 6),
        "vpp":            round(precision_score(y_true, y_pred, zero_division=0), 6),
        "vpn":            round(
                              tn / max(tn + fn, 1), 6
                          ),
        "f1":             round(f1_score(y_true, y_pred, zero_division=0), 6),
        "brier":          round(brier_score_loss(y_true, y_prob),         6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def evaluar():
    print("=" * 60)
    print("Experimento: Evaluación sobre test estratificado")
    print("=" * 60)

    test = pd.read_csv(DIR_DATA / "test_estratificado.csv")
    X_test, y_test = separar_x_y(test)
    print(f"test_estratificado: {len(test)} filas | "
          f"prevalencia {y_test.mean():.4f}")

    resultados = []

    for nombre_archivo, label in [
        ("modelo_xgboost_estratificado.pkl", "XGBoost"),
        ("modelo_forest_estratificado.pkl",  "Random Forest"),
    ]:
        ruta = DIR_MOD / nombre_archivo
        obj = joblib.load(ruta)
        modelo   = obj["modelo"]
        columnas = obj["columnas"]

        X_aligned = X_test.reindex(columns=columnas, fill_value=0)
        y_prob = modelo.predict_proba(X_aligned)[:, 1]

        m = calcular_metricas(label, y_test.values, y_prob)
        resultados.append(m)
        print(
            f"\n{label}: AUC-ROC {m['auc_roc']:.4f} | "
            f"AUC-PR {m['auc_pr']:.4f} | F1 {m['f1']:.4f} | "
            f"Sens {m['sensibilidad']:.4f} | Esp {m['especificidad']:.4f} | "
            f"Brier {m['brier']:.4f}"
        )

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(DIR_RESULTS / "metricas_estratificado.csv", index=False)
    print(f"\nGuardado: {DIR_RESULTS / 'metricas_estratificado.csv'}")

    df_prod = pd.read_csv(RUTA_PROD)
    print("\n\nTABLA COMPARATIVA — producción vs estratificado")
    print("=" * 60)

    metricas_cmp = ["auc_roc", "auc_pr", "f1", "sensibilidad", "especificidad", "brier"]
    filas_cmp = []

    for row_est in resultados:
        nombre = row_est["modelo"]
        row_prod = df_prod[df_prod["modelo"] == nombre]
        if row_prod.empty:
            print(f"  Advertencia: {nombre} no encontrado en tabla producción")
            continue
        row_prod = row_prod.iloc[0]

        for m in metricas_cmp:
            val_prod = row_prod[m]
            val_est  = row_est[m]
            delta    = val_est - val_prod
            filas_cmp.append({
                "modelo":        nombre,
                "metrica":       m,
                "split_produccion":   round(float(val_prod), 6),
                "split_estratificado": round(float(val_est),  6),
                "delta":         round(float(delta), 6),
            })
            print(
                f"  {nombre:<18} {m:<16} "
                f"prod={val_prod:.4f}  est={val_est:.4f}  "
                f"delta={delta:+.4f}"
            )

    df_cmp = pd.DataFrame(filas_cmp)
    df_cmp.to_csv(DIR_RESULTS / "comparativa_sensibilidad.csv", index=False)
    print(f"\nGuardado: {DIR_RESULTS / 'comparativa_sensibilidad.csv'}")

    delta_auc = df_cmp[df_cmp["metrica"] == "auc_roc"]["delta"].abs().max()
    delta_pr  = df_cmp[df_cmp["metrica"] == "auc_pr"]["delta"].abs().max()
    delta_f1  = df_cmp[df_cmp["metrica"] == "f1"]["delta"].abs().max()
    print(f"\n--- Delta máx AUC-ROC: {delta_auc:.4f} (umbral < 0.010)")
    print(f"--- Delta máx AUC-PR:  {delta_pr:.4f} (umbral < 0.010)")
    print(f"--- Delta máx F1:      {delta_f1:.4f} (umbral < 0.020)")
    if delta_auc < 0.01 and delta_pr < 0.01 and delta_f1 < 0.02:
        print("las conclusiones del benchmark no dependen del split.")
    else:
        print("al menos una métrica supera el umbral.")

    return df_res, df_cmp

if __name__ == "__main__":
    evaluar()
