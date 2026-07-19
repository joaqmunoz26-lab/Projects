
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_split = import_module("05_split")

DIR_BASE    = Path(__file__).resolve().parent.parent
DIR_REPORTS = DIR_BASE / "reports"
DIR_MODELOS = DIR_BASE / "models"

def calcular_beneficio_neto(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbral: float,
) -> float:
    if umbral <= 0 or umbral >= 1:
        return 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    y_pred = (y_proba >= umbral).astype(int)
    tp     = int(((y_pred == 1) & (y_true == 1)).sum())
    fp     = int(((y_pred == 1) & (y_true == 0)).sum())

    return float((tp / n) - (fp / n) * (umbral / (1.0 - umbral)))

def calcular_beneficio_neto_treat_all(y_true: np.ndarray, umbral: float) -> float:
    if umbral <= 0 or umbral >= 1:
        return 0.0
    prevalencia = float(y_true.mean())
    return prevalencia - (1.0 - prevalencia) * (umbral / (1.0 - umbral))

def cargar_predicciones_modelos(df_test: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    predicciones = {}

    candidatos = [
        ("XGBoost",             "modelo_xgboost.pkl",   None),
    ]

    for nombre, archivo, scaler_key in candidatos:
        ruta = DIR_MODELOS / archivo
        if not ruta.exists():
            print(f"  [omitido] {nombre}: {archivo} no encontrado")
            continue
        try:
            obj      = joblib.load(ruta)
            modelo   = obj["modelo"]
            columnas = obj["columnas"]
            X        = X_test.reindex(columns=columnas, fill_value=0)

            if scaler_key and scaler_key in obj:
                X_arr = obj[scaler_key].transform(X)
            else:
                X_arr = X.values

            predicciones[nombre] = modelo.predict_proba(X_arr)[:, 1]
            print(f"  [ok] {nombre}: {len(predicciones[nombre])} predicciones")
        except Exception as exc:
            print(f"  [error] {nombre}: {exc}")

    return predicciones

def ejecutar_dca(
    umbral_min: float = 0.05,
    umbral_max: float = 0.50,
    n_umbrales: int   = 226,
) -> tuple:
    print("=" * 60)
    print("Decision Curve Analysis (DCA) - Vickers & Elkin 2006")
    print("=" * 60)

    df_test = pd.read_csv(DIR_BASE / "data" / "processed" / "test.csv")
    X_test, y_test = _split.separar_x_y(df_test)
    y_arr = y_test.values

    print(f"\nTest set : {len(df_test)} filas")
    print(f"Prevalencia (descompensacion=1): {y_arr.mean():.2%}")

    print("\nCargando modelos...")
    predicciones = cargar_predicciones_modelos(df_test, X_test)

    if not predicciones:
        raise RuntimeError(
            "No se cargo ningun modelo. "
            "Ejecuta primero: python src/06a_train_ml.py --modelos xgboost"
        )

    umbrales  = np.linspace(umbral_min, umbral_max, n_umbrales)
    registros = []

    print("\nCalculando beneficio neto por umbral...")
    for umbral in umbrales:
        for nombre, y_proba in predicciones.items():
            y_t = y_arr
            registros.append({
                "estrategia":    nombre,
                "umbral":        round(float(umbral), 4),
                "beneficio_neto": round(calcular_beneficio_neto(y_t, y_proba, umbral), 6),
                "tipo":          "modelo",
            })

        registros.append({
            "estrategia":    "Tratar a todos",
            "umbral":        round(float(umbral), 4),
            "beneficio_neto": round(calcular_beneficio_neto_treat_all(y_arr, umbral), 6),
            "tipo":          "baseline",
        })
        registros.append({
            "estrategia":    "No tratar a nadie",
            "umbral":        round(float(umbral), 4),
            "beneficio_neto": 0.0,
            "tipo":          "baseline",
        })

    return pd.DataFrame(registros), predicciones, y_arr

def graficar_curvas_dca(df_resultados: pd.DataFrame, ruta_salida: Path) -> None:
    COLORES = {
        "Logistic Regression": "#378ADD",
        "Random Forest":       "#639922",
        "XGBoost":             "#D85A30",
        "Tratar a todos":      "#888780",
        "No tratar a nadie":   "#B4B2A9",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for estrategia in df_resultados["estrategia"].unique():
        sub = df_resultados[df_resultados["estrategia"] == estrategia].sort_values("umbral")
        es_baseline = sub["tipo"].iloc[0] == "baseline"
        ax.plot(
            sub["umbral"], sub["beneficio_neto"],
            "--" if es_baseline else "-",
            linewidth=1.5 if es_baseline else 2.2,
            color=COLORES.get(estrategia, "gray"),
            label=estrategia,
        )

    ax.axhline(y=0, color="k", linestyle=":", alpha=0.35, linewidth=0.8)
    ax.set_xlabel("Umbral de probabilidad (Threshold)", fontsize=11)
    ax.set_ylabel("Beneficio Neto (Net Benefit)", fontsize=11)
    ax.set_title(
        "Decision Curve Analysis\n"
        "Modelos vs alternativas triviales (tratar todos / tratar a nadie)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=130)
    plt.close(fig)
    print(f"  Plot guardado: {ruta_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decision Curve Analysis para modelos de la tesis"
    )
    parser.add_argument("--umbral-min", type=float, default=0.05)
    parser.add_argument("--umbral-max", type=float, default=0.50)
    parser.add_argument("--n-umbrales", type=int, default=226)
    args = parser.parse_args()

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    df_res, predicciones, y_test = ejecutar_dca(
        umbral_min=args.umbral_min,
        umbral_max=args.umbral_max,
        n_umbrales=args.n_umbrales,
    )

    ruta_csv = DIR_REPORTS / "dca_resultados.csv"
    df_res.to_csv(ruta_csv, index=False)
    print(f"\nResultados guardados: {ruta_csv}")

    graficar_curvas_dca(df_res, DIR_REPORTS / "dca_curva_comparativa.png")

    print("\n" + "=" * 60)
    print("Beneficio Neto en umbrales clave (umbral -> modelo vs treat-all)")
    print("=" * 60)
    for u in [0.10, 0.20, 0.30, 0.40]:
        sub = df_res[df_res["umbral"] == round(u, 4)]
        linea_partes = []
        for est in predicciones:
            nb = sub[sub["estrategia"] == est]["beneficio_neto"]
            if len(nb):
                linea_partes.append(f"{est}={float(nb.iloc[0]):.4f}")
        nb_all = sub[sub["estrategia"] == "Tratar a todos"]["beneficio_neto"]
        if len(nb_all):
            linea_partes.append(f"treat-all={float(nb_all.iloc[0]):.4f}")
        print(f"  p={u:.2f}: {' | '.join(linea_partes)}")
