
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DIR_BASE    = Path(__file__).resolve().parent.parent
DIR_REPORTS = DIR_BASE / "reports"

def ejecutar_cv_estratificado(
    df,
    target_col: str = "descompensacion_glicemica_90d",
    k: int = 5,
    modelos: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    modelos = modelos or ["logistic", "random_forest", "xgboost"]

    columnas_excluir = [
        "paciente_id", "fecha_control", "control_num",
        "prob_descompensacion_90d", target_col,
    ]
    columnas_excluir = [c for c in columnas_excluir if c in df.columns]

    y = df[target_col].astype(int)
    X = df.drop(columns=columnas_excluir)
    cols_cat = [c for c in ["sexo"] if c in X.columns]
    if cols_cat:
        X = pd.get_dummies(X, columns=cols_cat, drop_first=True)
    X = X.fillna(0)

    skf       = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    resultados = []

    for idx_m, nombre in enumerate(modelos):
        print(f"\n[{idx_m + 1}/{len(modelos)}] Cross-validation: {nombre}")

        for pliegue, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            if nombre == "logistic":
                sc   = StandardScaler()
                X_trs = sc.fit_transform(X_tr)
                X_tes = sc.transform(X_te)
                m = LogisticRegression(
                    max_iter=1000, C=0.1, penalty="l2",
                    class_weight="balanced", random_state=seed,
                )
                m.fit(X_trs, y_tr)
                proba = m.predict_proba(X_tes)[:, 1]

            elif nombre == "random_forest":
                m = RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    class_weight="balanced", n_jobs=-1, random_state=seed,
                )
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]

            elif nombre == "xgboost":
                scale_pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                m = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    missing=np.nan,
                    scale_pos_weight=scale_pw,
                    random_state=seed, eval_metric="auc",
                )
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)[:, 1]

            else:
                print(f"  {nombre} no soportado en CV, saltando.")
                continue

            pred = (proba >= 0.5).astype(int)
            resultados.append({
                "modelo":    nombre,
                "pliegue":   pliegue,
                "auc":       roc_auc_score(y_te, proba),
                "recall":    recall_score(y_te, pred, zero_division=0),
                "precision": precision_score(y_te, pred, zero_division=0),
                "f1":        f1_score(y_te, pred, zero_division=0),
                "n_train":   len(tr_idx),
                "n_test":    len(te_idx),
            })
            print(f"  Pliegue {pliegue}: AUC={resultados[-1]['auc']:.4f} "
                  f"Recall={resultados[-1]['recall']:.4f}")

    return pd.DataFrame(resultados)

def calcular_intervalos_confianza(df_cv: pd.DataFrame) -> pd.DataFrame:
    metricas = ["auc", "recall", "precision", "f1"]
    resumen  = []

    for modelo in df_cv["modelo"].unique():
        sub = df_cv[df_cv["modelo"] == modelo]
        k   = len(sub)
        for m in metricas:
            vals    = sub[m].values
            media   = vals.mean()
            std     = vals.std(ddof=1) if k > 1 else 0.0
            ic_d    = 1.96 * std / np.sqrt(k)
            resumen.append({
                "modelo":          modelo,
                "metrica":         m,
                "media":           round(media, 4),
                "std":             round(std, 4),
                "ic_95_inferior":  round(media - ic_d, 4),
                "ic_95_superior":  round(media + ic_d, 4),
                "n_pliegues":      k,
            })

    return pd.DataFrame(resumen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int,   default=5)
    parser.add_argument("--modelos", nargs="+",
                        default=["logistic", "random_forest", "xgboost"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"Cross-validation estratificado K={args.k}")
    print("Nota: complemento al split por paciente de 05_split.py")
    print("=" * 60)

    ruta = DIR_BASE / "data" / "processed" / "pacientes_con_features.csv"
    if not ruta.exists():
        sys.exit(f"Archivo no encontrado: {ruta}\nEjecutá primero: python src/04_features.py")

    df = pd.read_csv(ruta)
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    df_cv = ejecutar_cv_estratificado(df, k=args.k, modelos=args.modelos)

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    df_cv.to_csv(DIR_REPORTS / "cv_resultados.csv", index=False)

    df_ic = calcular_intervalos_confianza(df_cv)
    df_ic.to_csv(DIR_REPORTS / "cv_intervalos_confianza.csv", index=False)

    print("\n" + "=" * 60)
    print("Intervalos de confianza 95%")
    print("=" * 60)
    for modelo in df_ic["modelo"].unique():
        print(f"\n{modelo.upper()}:")
        for _, row in df_ic[df_ic["modelo"] == modelo].iterrows():
            print(f"  {row['metrica']:10s}: {row['media']:.4f}  "
                  f"IC [{row['ic_95_inferior']:.4f}, {row['ic_95_superior']:.4f}]")

    print(f"\nResultados guardados en: {DIR_REPORTS}")
