from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
)
from xgboost import XGBClassifier

DIR_EXP   = Path(__file__).resolve().parent
DIR_ROOT  = DIR_EXP.parent.parent.parent
RUTA_CSV  = DIR_ROOT / "data" / "processed" / "pacientes_con_features.csv"
DIR_PROD_MODELS = DIR_ROOT / "models"

SEMILLAS = list(range(42, 52))
OBJETIVO  = "descompensacion_glicemica_90d"

sys.path.insert(0, str(DIR_ROOT / "src"))
from importlib import import_module
FEATURES_MODELO = import_module("04_features").FEATURES_MODELO

XGB_PARAMS = dict(
    n_estimators=126,
    max_depth=4,
    learning_rate=0.011662890273931383,
    subsample=0.7301321323053057,
    colsample_bytree=0.7554709158757928,
    min_child_weight=3,
    reg_alpha=0.9384800715909529,
    reg_lambda=0.0013820379228636995,
    objective="binary:logistic",
    eval_metric="auc",
    missing=np.nan,
    enable_categorical=False,
    random_state=42,
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

def aplicar_corte_temporal(df: pd.DataFrame, fraccion_final: float = 0.25) -> pd.DataFrame:
    df = df.sort_values([OBJETIVO[:-3] if False else "paciente_id", "fecha_control"])
    df = df.sort_values(["paciente_id", "fecha_control"])
    resultado = []
    for _pid, grupo in df.groupby("paciente_id"):
        n = len(grupo)
        corte = int(n * (1 - fraccion_final))
        resultado.append(grupo.iloc[corte:])
    return pd.concat(resultado, ignore_index=True)

def split_aleatorio(df, seed, prop_train=0.70, prop_val=0.15):
    rng = np.random.default_rng(seed=seed)
    pacientes = rng.permutation(df["paciente_id"].unique())
    n = len(pacientes)
    n_train = int(n * prop_train)
    n_val   = int(n * prop_val)
    tr = df[df["paciente_id"].isin(pacientes[:n_train])].copy()
    va = df[df["paciente_id"].isin(pacientes[n_train:n_train + n_val])].copy()
    te = df[df["paciente_id"].isin(pacientes[n_train + n_val:])].copy()
    return tr, va, te

def split_estratificado(df, seed, prop_train=0.70, prop_val=0.15):
    rng = np.random.default_rng(seed=seed)
    etiqueta = (
        df.groupby("paciente_id")[OBJETIVO]
        .max().astype(int).rename("estrato")
    )
    pacs_pos = etiqueta[etiqueta == 1].index.values
    pacs_neg = etiqueta[etiqueta == 0].index.values

    def _cortar(pacs):
        n = len(pacs)
        n_tr = int(n * prop_train)
        n_va = int(n * prop_val)
        return pacs[:n_tr], pacs[n_tr:n_tr + n_va], pacs[n_tr + n_va:]

    pos_tr, pos_va, pos_te = _cortar(rng.permutation(pacs_pos))
    neg_tr, neg_va, neg_te = _cortar(rng.permutation(pacs_neg))

    def _filtrar(ids):
        return df[df["paciente_id"].isin(ids)].copy()

    tr = _filtrar(np.concatenate([pos_tr, neg_tr]))
    va = _filtrar(np.concatenate([pos_va, neg_va]))
    te = _filtrar(np.concatenate([pos_te, neg_te]))
    return tr, va, te

def verificar_leakage(tr, va, te):
    s_tr = set(tr["paciente_id"].unique())
    s_va = set(va["paciente_id"].unique())
    s_te = set(te["paciente_id"].unique())
    assert len(s_tr & s_va) == 0, "LEAKAGE train-val"
    assert len(s_tr & s_te) == 0, "LEAKAGE train-test"
    assert len(s_va & s_te) == 0, "LEAKAGE val-test"

def separar_x_y(df):
    df = df.copy()
    if "sexo" in df.columns and df["sexo"].dtype == object:
        df["sexo"] = (df["sexo"].astype(str).str.upper() == "M").astype(int)
    y = df[OBJETIVO].astype(int)
    X = df[FEATURES_MODELO].copy()
    return X, y

def entrenar_xgboost(X_tr, y_tr, X_va, y_va):
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    m = XGBClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m

def entrenar_forest(X_tr, y_tr, _X_va=None, _y_va=None):
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(X_tr, y_tr)
    return m

def calcular_metricas(modelo, X_te, y_te, columnas_train):
    X_aligned = X_te.reindex(columns=columnas_train, fill_value=0)
    y_prob = modelo.predict_proba(X_aligned)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    return {
        "auc_roc":       round(roc_auc_score(y_te, y_prob),          6),
        "auc_pr":        round(average_precision_score(y_te, y_prob), 6),
        "f1":            round(f1_score(y_te, y_pred, zero_division=0), 6),
        "precision":     round(precision_score(y_te, y_pred, zero_division=0), 6),
        "recall":        round(recall_score(y_te, y_pred, zero_division=0), 6),
        "sensibilidad":  round(tp / max(tp + fn, 1), 6),
        "especificidad": round(tn / max(tn + fp, 1), 6),
        "brier":         round(brier_score_loss(y_te, y_prob), 6),
    }

def ejecutar():
    print("=" * 65)
    print("Experimento multi semilla: split aleatorio vs estratificado")
    print(f"K={len(SEMILLAS)} semillas: {SEMILLAS}")
    print("=" * 65)

    df_full = pd.read_csv(RUTA_CSV)
    print(f"Dataset: {df_full.shape} | pacientes: {df_full['paciente_id'].nunique()}")

    filas = []

    for s in SEMILLAS:
        print(f"\n--- Semilla {s} ---")

        for estrategia, fn_split in [
            ("aleatorio",     split_aleatorio),
            ("estratificado", split_estratificado),
        ]:
            tr_raw, va_raw, te_raw = fn_split(df_full, seed=s)
            va_cut = aplicar_corte_temporal(va_raw)
            te_cut = aplicar_corte_temporal(te_raw)
            
            verificar_leakage(tr_raw, va_cut, te_cut)
            
            prev_tr = tr_raw[OBJETIVO].mean()
            prev_va = va_cut[OBJETIVO].mean()
            prev_te = te_cut[OBJETIVO].mean()

            X_tr, y_tr = separar_x_y(tr_raw)
            X_va, y_va = separar_x_y(va_cut)
            X_te, y_te = separar_x_y(te_cut)

            for nombre_m, fn_train in [
                ("XGBoost",       entrenar_xgboost),
                ("Random Forest", entrenar_forest),
            ]:
                modelo = fn_train(X_tr, y_tr, X_va, y_va)
                m = calcular_metricas(modelo, X_te, y_te, list(X_tr.columns))

                fila = {
                    "semilla":    s,
                    "estrategia": estrategia,
                    "modelo":     nombre_m,
                    "prev_train": round(prev_tr, 4),
                    "prev_val":   round(prev_va, 4),
                    "prev_test":  round(prev_te, 4),
                    **m,
                }
                filas.append(fila)
                print(
                    f"  [{estrategia[:3]}] {nombre_m:<14} "
                    f"AUC-ROC={m['auc_roc']:.4f}  "
                    f"AUC-PR={m['auc_pr']:.4f}  "
                    f"F1={m['f1']:.4f}"
                )

    df_res = pd.DataFrame(filas)
    ruta_out = DIR_EXP / "results" / "metricas_multiseed.csv"
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(ruta_out, index=False)
    print(f"\nGuardado: {ruta_out}")
    print(f"Total filas: {len(df_res)}")
    return df_res

if __name__ == "__main__":
    ejecutar()
