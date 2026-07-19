import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")

DIR_BASE = Path(__file__).resolve().parent.parent
DIR_MODELOS = DIR_BASE / "models"

def clasificar_riesgo(probabilidad: float) -> str:
    if probabilidad < 0.20:
        return "bajo"
    if probabilidad < 0.50:
        return "moderado"
    if probabilidad < 0.75:
        return "alto"
    return "critico"

def predecir_paciente(historial: pd.DataFrame) -> dict:
    obj = joblib.load(DIR_MODELOS / "modelo_xgboost.pkl")
    modelo = obj["modelo"]
    columnas_modelo = obj["columnas"]

    if len(historial) == 0:
        raise ValueError("Historial vacio")

    ultimo = historial.tail(1).copy()

    if "descompensacion_glicemica_90d" not in ultimo.columns:
        ultimo["descompensacion_glicemica_90d"] = 0
    if "prob_descompensacion_90d" not in ultimo.columns:
        ultimo["prob_descompensacion_90d"] = 0.0

    X, _ = _split.separar_x_y(ultimo)
    X = X.reindex(columns=columnas_modelo, fill_value=0)

    probabilidad = float(modelo.predict_proba(X)[0, 1])
    riesgo = clasificar_riesgo(probabilidad)

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)[0]

    factores = pd.DataFrame({
        "feature": columnas_modelo,
        "valor": X.iloc[0].values,
        "contribucion_shap": shap_values,
    })
    factores["abs_shap"] = np.abs(factores["contribucion_shap"])
    factores = factores.sort_values("abs_shap", ascending=False).head(5)

    top_factores = []
    for _, row in factores.iterrows():
        top_factores.append({
            "feature": row["feature"],
            "valor": round(float(row["valor"]), 3),
            "contribucion": round(float(row["contribucion_shap"]), 4),
            "direccion": "aumenta_riesgo" if row["contribucion_shap"] > 0 else "baja_riesgo",
        })

    return {
        "paciente_id": ultimo.get("paciente_id", pd.Series(["?"])).iloc[0],
        "fecha_control": str(ultimo.get("fecha_control", pd.Series(["?"])).iloc[0]),
        "probabilidad_descompensacion_90d": round(probabilidad, 4),
        "clasificacion_riesgo": riesgo,
        "top_factores": top_factores,
    }

def generar_waterfall_shap(
    paciente_id: str = None,
    datos_paciente: dict = None,
    modelo_pkl: str = "modelo_xgboost.pkl",
    ruta_salida: Path = None,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = joblib.load(DIR_MODELOS / modelo_pkl)
    modelo   = obj["modelo"]
    columnas = obj["columnas"]
    scaler   = obj.get("scaler", None)

    if paciente_id is not None:
        df_test = pd.read_csv(DIR_BASE / "data" / "processed" / "test.csv")
        historial = df_test[df_test["paciente_id"] == paciente_id]
        if len(historial) == 0:
            raise ValueError(f"Paciente {paciente_id} no encontrado en test.csv")
        ultimo = historial.tail(1).copy()
        if "descompensacion_glicemica_90d" not in ultimo.columns:
            ultimo["descompensacion_glicemica_90d"] = 0
        X_paciente, _ = _split.separar_x_y(ultimo)
        X_paciente = X_paciente.reindex(columns=columnas, fill_value=0)
    else:
        X_paciente = pd.DataFrame([datos_paciente])
        X_paciente = X_paciente.reindex(columns=columnas, fill_value=0)

    if scaler is not None:
        X_para_shap = pd.DataFrame(
            scaler.transform(X_paciente), columns=X_paciente.columns
        )
        bg = np.zeros((1, len(columnas)))
        explainer = shap.LinearExplainer(modelo, bg)
    else:
        X_para_shap = X_paciente
        explainer   = shap.TreeExplainer(modelo)

    shap_values = explainer.shap_values(X_para_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    if ruta_salida is None:
        nombre = f"shap_waterfall_{paciente_id or 'custom'}.png"
        ruta_salida = DIR_BASE / "reports" / "shap_individual" / nombre
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=float(base_value),
        data=X_paciente.iloc[0].values,
        feature_names=list(X_paciente.columns),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
    plt.close("all")

    probabilidad = float(modelo.predict_proba(X_para_shap)[0, 1])
    contribuciones = pd.DataFrame({
        "feature":    columnas,
        "valor":      X_paciente.iloc[0].values,
        "shap_value": shap_values[0],
    })
    contribuciones["abs_shap"] = contribuciones["shap_value"].abs()

    positivos = contribuciones[contribuciones["shap_value"] > 0].nlargest(5, "abs_shap")
    negativos = contribuciones[contribuciones["shap_value"] < 0].nlargest(5, "abs_shap")

    return {
        "paciente_id":                   paciente_id or "custom",
        "riesgo_predicho":               round(probabilidad, 4),
        "base_value":                    round(float(base_value), 4),
        "shap_sum":                      round(float(shap_values[0].sum()), 4),
        "ruta_plot":                     str(ruta_salida),
        "top_5_factores_aumentan_riesgo": positivos[["feature", "valor", "shap_value"]].to_dict("records"),
        "top_5_factores_reducen_riesgo":  negativos[["feature", "valor", "shap_value"]].to_dict("records"),
    }

def predecir_desde_csv(paciente_id: str):
    df_test = pd.read_csv(DIR_BASE / "data/processed/test.csv")
    historial = df_test[df_test["paciente_id"] == paciente_id]

    if len(historial) == 0:
        print(f"Paciente {paciente_id} no encontrado en test set")
        ejemplos = df_test["paciente_id"].unique()[:5]
        print(f"Ejemplos disponibles: {list(ejemplos)}")
        return None

    return predecir_paciente(historial)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paciente", type=str, default=None,
                        help="ID del paciente (ej: P00001)")
    args = parser.parse_args()

    print("=" * 60)
    print("Prediccion individual")
    print("=" * 60)

    if args.paciente is None:
        df_test = pd.read_csv(DIR_BASE / "data/processed/test.csv")
        pid = df_test["paciente_id"].iloc[0]
        print(f"(sin --paciente, usando ejemplo: {pid})")
    else:
        pid = args.paciente

    resultado = predecir_desde_csv(pid)

    if resultado:
        print(f"\nPaciente: {resultado['paciente_id']}")
        print(f"Fecha control: {resultado['fecha_control']}")
        print(f"\nProbabilidad descompensacion 90d: "
              f"{resultado['probabilidad_descompensacion_90d']:.2%}")
        print(f"Clasificacion: {resultado['clasificacion_riesgo'].upper()}")
        print("\nTop 5 factores:")
        for f in resultado["top_factores"]:
            signo = "+" if f["contribucion"] > 0 else ""
            print(f"  {f['feature']:40s} = {f['valor']:>10.3f}  "
                  f"SHAP={signo}{f['contribucion']:.4f}  [{f['direccion']}]")
