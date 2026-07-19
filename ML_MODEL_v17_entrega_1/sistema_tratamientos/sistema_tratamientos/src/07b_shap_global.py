import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")

DIR_BASE    = Path(__file__).resolve().parent.parent
DIR_MODELOS = DIR_BASE / "models"
DIR_REPORTS = DIR_BASE / "reports" / "shap_individual"
OUT_CSV     = DIR_REPORTS / "shap_global_importancia.csv"

def calcular_importancia_global(
    modelo_pkl: str = "modelo_xgboost.pkl",
    ruta_test: Path = None,
) -> pd.DataFrame:
    ruta_test = Path(ruta_test) if ruta_test else DIR_BASE / "data" / "processed" / "test.csv"

    obj      = joblib.load(DIR_MODELOS / modelo_pkl)
    modelo   = obj["modelo"]
    columnas = obj["columnas"]
    scaler   = obj.get("scaler", None)

    df_test = pd.read_csv(ruta_test)
    X, _    = _split.separar_x_y(df_test)
    X       = X.reindex(columns=columnas, fill_value=0)
    X_shap  = scaler.transform(X) if scaler is not None else X

    explainer   = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)

    abs_shap  = np.abs(shap_values)
    media_abs = abs_shap.mean(axis=0)
    total     = media_abs.sum()

    return pd.DataFrame({
        "feature":        list(columnas),
        "media_abs_shap": media_abs,
        "pct":            100.0 * media_abs / total if total > 0 else media_abs,
    }).sort_values("media_abs_shap", ascending=False).reset_index(drop=True)

def main() -> Path:
    print("=" * 60)
    print("Importancia SHAP global sobre el conjunto de prueba")
    print("=" * 60)

    ruta_test = DIR_BASE / "data" / "processed" / "test.csv"
    df_test   = pd.read_csv(ruta_test)
    n_pac     = df_test["paciente_id"].nunique()
    n_ctrl    = len(df_test)

    df_imp = calcular_importancia_global(ruta_test=ruta_test)

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(OUT_CSV, index=False)

    print(f"\nMuestra: {n_pac} pacientes ({n_ctrl} controles) del conjunto de prueba")
    print("\nRanking de importancia (media |SHAP|):")
    for _, r in df_imp.iterrows():
        print(f"  {r['feature']:28s} {r['pct']:5.1f}%")

    top3 = df_imp.head(3)
    print(f"\nTop-3: {list(top3['feature'])}")
    print(f"Concentracion top-3: {top3['pct'].sum():.1f}%")
    print(f"\nCSV: {OUT_CSV}")
    return OUT_CSV

if __name__ == "__main__":
    main()
