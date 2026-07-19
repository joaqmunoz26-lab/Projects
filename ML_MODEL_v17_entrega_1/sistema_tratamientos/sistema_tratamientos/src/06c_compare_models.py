
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")
_baseline = import_module("06b_baseline_clinico")

DIR_BASE = Path(__file__).resolve().parent.parent
DIR_MODELOS = DIR_BASE / "models"
DIR_REPORTS = DIR_BASE / "reports"

COLORES = {
    "Regla simple": "#B4B2A9",
    "Regla extendida": "#888780",
    "Logistic Regression": "#378ADD",
    "Elastic Net": "#85B7EB",
    "Decision Tree": "#97C459",
    "Random Forest": "#639922",
    "XGBoost": "#D85A30",
    "LightGBM": "#EF9F27",
}

MODELOS_OFICIALES = {"Logistic Regression", "Random Forest", "XGBoost", "LightGBM"}

def calcular_metricas(y_true, y_proba, y_pred=None, umbral=0.5):
    if y_pred is None:
        y_pred = (y_proba >= umbral).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
        "sensibilidad": tp / max(tp + fn, 1),
        "especificidad": tn / max(tn + fp, 1),
        "vpp": tp / max(tp + fp, 1),
        "vpn": tn / max(tn + fn, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def predecir_modelo_sklearn(obj, X):
    modelo = obj["modelo"]
    X = X.reindex(columns=obj["columnas"], fill_value=0)
    if "scaler" in obj:
        X_arr = obj["scaler"].transform(X)
    else:
        X_arr = X.values
    return modelo.predict_proba(X_arr)[:, 1]

def cargar_predicciones_todos(df_test, X_test, y_test):
    predicciones = {}

    predicciones["Regla simple"] = _baseline.regla_simple_probabilistica(df_test).values
    predicciones["Regla extendida"] = _baseline.regla_extendida(df_test)

    mapeo_modelos = [
        ("Logistic Regression", "modelo_logistic.pkl"),
        ("Elastic Net",         "modelo_elasticnet.pkl"),
        ("Decision Tree",       "modelo_tree.pkl"),
        ("Random Forest",       "modelo_forest.pkl"),
        ("XGBoost",             "modelo_xgboost.pkl"),
        ("LightGBM",            "modelo_lightgbm.pkl"),
    ]

    for nombre, archivo in mapeo_modelos:
        ruta = DIR_MODELOS / archivo
        if not ruta.exists():
            print(f"  (saltando {nombre}: no se encontro {archivo})")
            continue
        try:
            obj = joblib.load(ruta)
            predicciones[nombre] = predecir_modelo_sklearn(obj, X_test)
            print(f"  cargado: {nombre}")
        except Exception as e:
            print(f"  error cargando {nombre}: {e}")

    return predicciones

def grafico_roc_comparativo(y_test, predicciones, ruta):
    plt.figure(figsize=(8, 7))
    for nombre, proba in predicciones.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        estilo = "--" if "Regla" in nombre else "-"
        ancho = 1.5 if "Regla" in nombre else 2.0
        plt.plot(fpr, tpr, estilo, linewidth=ancho,
                 color=COLORES.get(nombre, "gray"),
                 label=f"{nombre} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=1, label="Aleatorio")
    plt.xlabel("1 - Especificidad (FPR)", fontsize=11)
    plt.ylabel("Sensibilidad (TPR)", fontsize=11)
    plt.title("Comparativa de modelos - Curva ROC", fontsize=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def grafico_pr_comparativo(y_test, predicciones, ruta):
    plt.figure(figsize=(8, 7))
    for nombre, proba in predicciones.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        estilo = "--" if "Regla" in nombre else "-"
        ancho = 1.5 if "Regla" in nombre else 2.0
        plt.plot(rec, prec, estilo, linewidth=ancho,
                 color=COLORES.get(nombre, "gray"),
                 label=f"{nombre} (AP={ap:.3f})")

    tasa = y_test.mean()
    plt.axhline(y=tasa, color="k", linestyle=":", alpha=0.4,
                label=f"Prevalencia ({tasa:.2f})")
    plt.xlabel("Recall (Sensibilidad)", fontsize=11)
    plt.ylabel("Precision (VPP)", fontsize=11)
    plt.title("Comparativa de modelos - Curva Precision-Recall", fontsize=12)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def grafico_calibracion_comparativo(y_test, predicciones, ruta):
    plt.figure(figsize=(8, 7))
    for nombre, proba in predicciones.items():
        if "Regla" in nombre:
            continue
        try:
            prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
            plt.plot(prob_pred, prob_true, "o-", linewidth=2, markersize=5,
                     color=COLORES.get(nombre, "gray"),
                     label=f"{nombre} (Brier={brier_score_loss(y_test, proba):.3f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Calibracion perfecta")
    plt.xlabel("Probabilidad predicha", fontsize=11)
    plt.ylabel("Frecuencia observada", fontsize=11)
    plt.title("Comparativa de modelos - Calibracion", fontsize=12)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def grafico_ranking(tabla, ruta):
    tabla_sorted = tabla.sort_values("auc_roc", ascending=True)
    colores = [COLORES.get(m, "gray") for m in tabla_sorted["modelo"]]

    plt.figure(figsize=(9, max(4, len(tabla_sorted) * 0.5)))
    barras = plt.barh(tabla_sorted["modelo"], tabla_sorted["auc_roc"],
                      color=colores, edgecolor="white", linewidth=1)

    for i, (_, auc) in enumerate(zip(barras, tabla_sorted["auc_roc"])):
        plt.text(auc + 0.005, i, f"{auc:.3f}",
                 va="center", fontsize=10)

    plt.axvline(x=0.5, color="k", linestyle=":", alpha=0.3, label="Aleatorio")
    plt.xlabel("AUC-ROC (test set)", fontsize=11)
    plt.title("Ranking de modelos por AUC-ROC", fontsize=12)
    plt.xlim(0, 1.02)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def comparar_modelos(umbral: float = 0.5):
    print("=" * 60)
    print("Comparativa de modelos")
    print("=" * 60)

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(DIR_BASE / "data/processed/test.csv")
    X_test, y_test = _split.separar_x_y(df_test)

    print(f"\nTest: {len(df_test)} filas, {y_test.sum()} eventos ({y_test.mean():.2%})")
    print("\nCargando modelos disponibles...")
    predicciones = cargar_predicciones_todos(df_test, X_test, y_test)

    if len(predicciones) < 3:
        print("\nError: menos de 3 modelos disponibles. Ejecuta primero 06a_train_ml.py")
        return

    print(f"\nCalculando metricas para {len(predicciones)} modelos...")
    filas = []
    for nombre, proba in predicciones.items():
        metricas = calcular_metricas(y_test.values, proba, umbral=umbral)
        filas.append({"modelo": nombre, **metricas})

    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values("auc_roc", ascending=False)

    tabla.to_csv(DIR_REPORTS / "tabla_comparativa.csv", index=False)
    tabla.to_csv(DIR_REPORTS / "tabla_comparativa_exploratoria.csv", index=False)

    tabla_oficial = tabla[tabla["modelo"].isin(MODELOS_OFICIALES)]
    tabla_oficial.to_csv(DIR_REPORTS / "tabla_comparativa_oficial.csv", index=False)

    print("\nGenerando graficos comparativos...")
    grafico_roc_comparativo(y_test, predicciones, DIR_REPORTS / "roc_comparativa.png")
    grafico_pr_comparativo(y_test, predicciones, DIR_REPORTS / "pr_comparativa.png")
    grafico_calibracion_comparativo(y_test, predicciones, DIR_REPORTS / "calibracion_comparativa.png")
    grafico_ranking(tabla, DIR_REPORTS / "ranking_modelos.png")

    print("\n" + "=" * 60)
    print("Ranking por AUC-ROC en test set")
    print("=" * 60)
    for _, fila in tabla.iterrows():
        print(f"  {fila['modelo']:22s}: AUC={fila['auc_roc']:.3f}  "
              f"Sens={fila['sensibilidad']:.3f}  Esp={fila['especificidad']:.3f}  "
              f"Brier={fila['brier']:.3f}")

    mejor = tabla.iloc[0]
    print("\n" + "=" * 60)
    print(f"Mejor modelo: {mejor['modelo']}")
    print("=" * 60)
    print(f"  AUC-ROC:       {mejor['auc_roc']:.4f}")
    print(f"  AUC-PR:        {mejor['auc_pr']:.4f}")
    print(f"  Sensibilidad:  {mejor['sensibilidad']:.4f}")
    print(f"  Especificidad: {mejor['especificidad']:.4f}")
    print(f"  Brier:         {mejor['brier']:.4f}")

    mejor_ml = tabla[~tabla["modelo"].str.contains("Regla")].iloc[0]
    mejor_baseline = tabla[tabla["modelo"].str.contains("Regla")].iloc[0]
    mejora = mejor_ml["auc_roc"] - mejor_baseline["auc_roc"]
    print("\nMejora del mejor ML sobre mejor baseline clinico:")
    print(f"  {mejor_ml['modelo']} vs {mejor_baseline['modelo']}: +{mejora:.3f} AUC")

    print(f"\nArchivos generados en: {DIR_REPORTS}")
    print("  - tabla_comparativa_oficial.csv    (benchmark tesis: 3 modelos)")
    print("  - tabla_comparativa_exploratoria.csv (6 modelos: para anexo)")
    print("  - tabla_comparativa.csv            (todos los modelos)")
    print("  - roc_comparativa.png")
    print("  - pr_comparativa.png")
    print("  - calibracion_comparativa.png")
    print("  - ranking_modelos.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="Umbral de decision para metricas binarias")
    args = parser.parse_args()
    comparar_modelos(args.umbral)
