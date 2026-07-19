
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")

DIR_BASE = Path(__file__).resolve().parent.parent
DIR_MODELOS = DIR_BASE / "models"
DIR_REPORTS = DIR_BASE / "reports"

def metricas_principales(y_true, y_proba, umbral=0.5):
    y_pred = (y_proba >= umbral).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "vpp_precision": vpp,
        "vpn": vpn,
        "brier_score": brier_score_loss(y_true, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def evaluar_matriz_confusion(y_true, y_proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * vpp * sensibilidad / (vpp + sensibilidad)
          if (vpp + sensibilidad) > 0 else 0.0)
    f2 = (5 * vpp * sensibilidad / (4 * vpp + sensibilidad)
          if (4 * vpp + sensibilidad) > 0 else 0.0)

    return {
        "threshold": round(float(threshold), 4),
        "vp": int(tp), "fn": int(fn), "fp": int(fp), "vn": int(tn),
        "sensibilidad": sensibilidad, "especificidad": especificidad,
        "vpp": vpp, "vpn": vpn, "f1": f1, "f2": f2,
    }

def grafico_roc(y_true, y_proba_ml, y_proba_baseline, ruta_salida):
    fpr_ml, tpr_ml, _ = roc_curve(y_true, y_proba_ml)
    fpr_bl, tpr_bl, _ = roc_curve(y_true, y_proba_baseline)

    auc_ml = roc_auc_score(y_true, y_proba_ml)
    auc_bl = roc_auc_score(y_true, y_proba_baseline)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_ml, tpr_ml, label=f"XGBoost (AUC={auc_ml:.3f})", linewidth=2)
    plt.plot(fpr_bl, tpr_bl, label=f"Baseline clinico (AUC={auc_bl:.3f})",
             linewidth=2, linestyle="--")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Aleatorio")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=120)
    plt.close()

def grafico_calibracion(y_true, y_proba, ruta_salida):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "o-", label="Modelo", linewidth=2)
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Calibracion perfecta")
    plt.xlabel("Probabilidad predicha (media por bin)")
    plt.ylabel("Frecuencia observada")
    plt.title(f"Calibracion (Brier={brier_score_loss(y_true, y_proba):.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=120)
    plt.close()

def grafico_precision_recall(y_true, y_proba, ruta_salida):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, linewidth=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precision (VPP)")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=120)
    plt.close()

def analisis_shap(modelo, X_test, ruta_salida):
    print("  calculando SHAP values...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test.iloc[:500])

    plt.figure()
    shap.summary_plot(shap_values, X_test.iloc[:500],
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
    plt.close()

    return pd.DataFrame({
        "feature": X_test.columns,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False)

def analisis_subgrupos(df_test, y_proba):
    df = df_test.copy()
    df["proba"] = y_proba

    resultados = []

    df["grupo_edad"] = pd.cut(df["edad"], bins=[40, 55, 70, 80],
                              labels=["40-55", "55-70", "70-80"])

    for grupo, sub in df.groupby("grupo_edad", observed=True):
        if len(sub) < 30 or sub["descompensacion_glicemica_90d"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["descompensacion_glicemica_90d"], sub["proba"])
        resultados.append({
            "subgrupo": f"edad_{grupo}", "n": len(sub), "auc": auc,
        })

    for sexo, sub in df.groupby("sexo", observed=True):
        if sub["descompensacion_glicemica_90d"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["descompensacion_glicemica_90d"], sub["proba"])
        resultados.append({
            "subgrupo": f"sexo_{sexo}", "n": len(sub), "auc": auc,
        })

    return pd.DataFrame(resultados)

def evaluar():
    print("=" * 60)
    print("Evaluacion del modelo final")
    print("=" * 60)

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    obj_xgb = joblib.load(DIR_MODELOS / "modelo_xgboost.pkl")
    modelo = obj_xgb["modelo"]
    columnas = obj_xgb["columnas"]

    df_test = pd.read_csv(DIR_BASE / "data/processed/test.csv")
    X_test, y_test = _split.separar_x_y(df_test)
    X_test = X_test.reindex(columns=columnas, fill_value=0)

    print(f"\nTest: {len(df_test)} filas, {y_test.sum()} eventos ({y_test.mean():.2%})")

    proba = modelo.predict_proba(X_test)[:, 1]

    print("\n1. Metricas principales (umbral=0.5):")
    metricas = metricas_principales(y_test, proba, umbral=0.5)
    for k, v in metricas.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    baseline = pd.read_csv(DIR_MODELOS / "baseline_predicciones.csv")
    proba_baseline = baseline["proba_regla_extendida"].values[:len(y_test)]

    print("\n2. Generando graficos...")
    grafico_roc(y_test, proba, proba_baseline, DIR_REPORTS / "roc_curve.png")
    grafico_calibracion(y_test, proba, DIR_REPORTS / "calibracion.png")
    grafico_precision_recall(y_test, proba, DIR_REPORTS / "precision_recall.png")
    print(f"  graficos guardados en {DIR_REPORTS}")

    print("\n3. Analisis SHAP...")
    importancia = analisis_shap(modelo, X_test, DIR_REPORTS / "shap_summary.png")
    importancia.to_csv(DIR_REPORTS / "feature_importance.csv", index=False)
    print("\n  Top 10 features mas importantes:")
    print(importancia.head(10).to_string(index=False))

    print("\n4. Analisis de subgrupos...")
    subgrupos = analisis_subgrupos(df_test, proba)
    print(subgrupos.to_string(index=False))
    subgrupos.to_csv(DIR_REPORTS / "subgrupos.csv", index=False)

    pd.DataFrame([metricas]).to_csv(DIR_REPORTS / "metricas_finales.csv", index=False)
    print(f"\nTodo guardado en: {DIR_REPORTS}")

if __name__ == "__main__":
    evaluar()
