
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DIR_BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = DIR_BASE / "data" / "processed"
DIR_MODELOS = DIR_BASE / "models"

def regla_simple(df: pd.DataFrame) -> np.ndarray:
    return (df["hba1c"] > 8.0).astype(int).values

def regla_simple_probabilistica(df: pd.DataFrame) -> np.ndarray:
    z = (df["hba1c"] - 8.0) * 1.5
    return 1.0 / (1.0 + np.exp(-z))

def regla_extendida(df: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(df))

    score += np.clip((df["hba1c"].values - 6.5) / 3.0, 0, 1) * 0.45
    score += (1 - df["adherencia_tratamiento"].values) * 0.25
    score += (df["microalbuminuria"].values >= 30).astype(float) * 0.10
    score += df["ecg_anormal"].values.astype(float) * 0.05
    score += np.clip((60 - df["funcion_renal_egfr"].values) / 40, 0, 1) * 0.10
    score += df["hospitalizacion_previa_12m"].values.astype(float) * 0.05

    return np.clip(score, 0.0, 1.0)

def regla_extendida_binaria(df: pd.DataFrame, umbral: float = 0.5) -> np.ndarray:
    return (regla_extendida(df) >= umbral).astype(int)

def evaluar_baseline(nombre: str, y_true, y_pred_binario, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred_binario, zero_division=0)
    recall = recall_score(y_true, y_pred_binario, zero_division=0)
    f1 = f1_score(y_true, y_pred_binario, zero_division=0)

    print(f"\n  {nombre}:")
    print(f"    AUC:         {auc:.4f}")
    print(f"    Precision:   {precision:.4f}")
    print(f"    Recall:      {recall:.4f}")
    print(f"    F1:          {f1:.4f}")

    return {
        "baseline": nombre,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def ejecutar_baselines():
    print("=" * 60)
    print("Baselines clinicos")
    print("=" * 60)

    df_val = pd.read_csv(DIR_DATOS / "val.csv")
    df_test = pd.read_csv(DIR_DATOS / "test.csv")

    y_val = df_val["descompensacion_glicemica_90d"].values
    y_test = df_test["descompensacion_glicemica_90d"].values

    resultados = []

    print("\n--- Validation ---")
    resultados.append(evaluar_baseline(
        "Regla simple (HbA1c > 8.0) - VAL",
        y_val,
        regla_simple(df_val),
        regla_simple_probabilistica(df_val),
    ))
    resultados.append(evaluar_baseline(
        "Regla extendida - VAL",
        y_val,
        regla_extendida_binaria(df_val),
        regla_extendida(df_val),
    ))

    print("\n--- Test ---")
    resultados.append(evaluar_baseline(
        "Regla simple (HbA1c > 8.0) - TEST",
        y_test,
        regla_simple(df_test),
        regla_simple_probabilistica(df_test),
    ))
    resultados.append(evaluar_baseline(
        "Regla extendida - TEST",
        y_test,
        regla_extendida_binaria(df_test),
        regla_extendida(df_test),
    ))

    pd.DataFrame(resultados).to_csv(
        DIR_MODELOS / "baseline_resultados.csv", index=False
    )

    predicciones_test = pd.DataFrame({
        "paciente_id": df_test["paciente_id"],
        "fecha_control": df_test["fecha_control"],
        "y_true": y_test,
        "pred_regla_simple": regla_simple(df_test),
        "proba_regla_simple": regla_simple_probabilistica(df_test),
        "pred_regla_extendida": regla_extendida_binaria(df_test),
        "proba_regla_extendida": regla_extendida(df_test),
    })
    predicciones_test.to_csv(DIR_MODELOS / "baseline_predicciones.csv", index=False)

    print(f"\nResultados guardados en: {DIR_MODELOS}")

if __name__ == "__main__":
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    ejecutar_baselines()
