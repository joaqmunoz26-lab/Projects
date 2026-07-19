import sys
from pathlib import Path
from unittest.mock import MagicMock

_MOCKS = [
    "mlflow", "mlflow.sklearn",
    "optuna", "optuna.samplers", "optuna.logging",
    "xgboost",
    "lightgbm",
    "sklearn.frozen",
]
_SAVED_MODULES = {
    m: (None if isinstance(sys.modules.get(m), MagicMock) else sys.modules.get(m))
    for m in _MOCKS
}
for _m in _MOCKS:
    sys.modules[_m] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

_train = import_module("06a_train_ml")
_cv    = import_module("06d_cross_validation")

ejecutar_cv_estratificado     = _cv.ejecutar_cv_estratificado
calcular_intervalos_confianza = _cv.calcular_intervalos_confianza

@pytest.fixture(scope="module", autouse=True)
def _restore_mocks():
    yield
    for _m, original in _SAVED_MODULES.items():
        if original is None:
            sys.modules.pop(_m, None)
        else:
            sys.modules[_m] = original

def test_xgboost_tiene_missing_nan_declarado():
    import inspect
    src_objetivo = inspect.getsource(_train._objetivo_xgboost)
    src_entrenar = inspect.getsource(_train.entrenar_xgboost)
    assert "missing" in src_objetivo and "np.nan" in src_objetivo
    assert "missing" in src_entrenar and "np.nan" in src_entrenar

@pytest.fixture
def df_cv_plano():
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame({
        "feat1":  rng.normal(0, 1, n),
        "feat2":  rng.normal(0, 1, n),
        "target": (rng.random(n) < 0.20).astype(int),
    })

def test_cv_estratificado_mantiene_proporcion_clases(df_cv_plano):
    from sklearn.model_selection import StratifiedKFold

    prop_global = df_cv_plano["target"].mean()
    y = df_cv_plano["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _, te_idx in skf.split(df_cv_plano, y):
        prop_pliegue = y[te_idx].mean()
        assert abs(prop_pliegue - prop_global) < 0.10

    df_res = ejecutar_cv_estratificado(
        df_cv_plano, target_col="target", k=5, modelos=["logistic"]
    )
    assert len(df_res) == 5
    assert (df_res["auc"] > 0).all()

def test_cv_genera_k_pliegues(df_cv_plano):
    k = 3
    modelos = ["logistic", "random_forest"]
    df_res = ejecutar_cv_estratificado(
        df_cv_plano, target_col="target", k=k, modelos=modelos
    )
    for modelo in modelos:
        filas = df_res[df_res["modelo"] == modelo]
        assert len(filas) == k, f"{modelo}: esperado {k}, got {len(filas)}"

def test_intervalos_confianza_calculados_correctamente():
    k = 5
    valores_auc = [0.70, 0.72, 0.74, 0.68, 0.76]
    df_input = pd.DataFrame({
        "modelo":    ["logistic"] * k,
        "pliegue":   list(range(1, k + 1)),
        "auc":       valores_auc,
        "recall":    [0.5] * k,
        "precision": [0.5] * k,
        "f1":        [0.5] * k,
        "n_train":   [160] * k,
        "n_test":    [40]  * k,
    })
    df_ic = calcular_intervalos_confianza(df_input)
    fila = df_ic[(df_ic["modelo"] == "logistic") & (df_ic["metrica"] == "auc")].iloc[0]

    media_esp = np.mean(valores_auc)
    std_esp   = np.std(valores_auc, ddof=1)
    ic_d_esp  = 1.96 * std_esp / np.sqrt(k)

    assert fila["media"]          == pytest.approx(media_esp,            abs=1e-4)
    assert fila["ic_95_inferior"] == pytest.approx(media_esp - ic_d_esp, abs=1e-4)
    assert fila["ic_95_superior"] == pytest.approx(media_esp + ic_d_esp, abs=1e-4)
