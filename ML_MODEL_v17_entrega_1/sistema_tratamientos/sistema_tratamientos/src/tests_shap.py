import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

_MOCKS = [
    "mlflow", "mlflow.sklearn",
    "optuna", "optuna.samplers", "optuna.logging",
    "xgboost", "lightgbm",
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

_predict = import_module("08_predict")
generar_waterfall_shap      = _predict.generar_waterfall_shap
calcular_importancia_global = import_module("07b_shap_global").calcular_importancia_global
_FEATS12 = import_module("04_features").FEATURES_MODELO

@pytest.fixture(scope="module", autouse=True)
def _restore_mocks():
    yield
    for _m, original in _SAVED_MODULES.items():
        if original is None:
            sys.modules.pop(_m, None)
        else:
            sys.modules[_m] = original


FEATURES_TEST   = ["hba1c", "glucosa", "edad"]
DATOS_PACIENTE  = {"hba1c": 8.5, "glucosa": 185.0, "edad": 55.0}
N_FEATS         = len(FEATURES_TEST)
SHAP_VALS       = np.array([[0.10, 0.05, -0.05]])
BASE_VALUE      = 0.20

def _waterfall_stub(explanation, max_display=10, show=True):
    pass

@pytest.fixture
def mock_pkl():
    modelo = MagicMock()
    modelo.predict_proba.return_value = np.array([[0.70, 0.30]])
    return {"modelo": modelo, "columnas": FEATURES_TEST}

@pytest.fixture
def mock_explainer():
    exp = MagicMock()
    exp.shap_values.return_value = SHAP_VALS.copy()
    exp.expected_value = BASE_VALUE
    return exp

@pytest.fixture
def patched(mock_pkl, mock_explainer):
    with patch("joblib.load", return_value=mock_pkl):
        with patch("shap.TreeExplainer", return_value=mock_explainer):
            with patch("shap.plots.waterfall", side_effect=_waterfall_stub):
                yield

@pytest.fixture
def fake_test_csv(tmp_path):
    rng = np.random.default_rng(42)
    n   = 20
    df  = pd.DataFrame({
        "paciente_id":                  [f"P{i:04d}" for i in range(n)],
        "fecha_control":                ["2024-01-01"] * n,
        "hba1c":                        rng.uniform(6.5, 12.0, n),
        "presion_sistolica":            rng.uniform(110, 180, n),
        "presion_diastolica":           rng.uniform(70, 110, n),
        "funcion_renal_egfr":           rng.uniform(40, 110, n),
        "colesterol_ldl":               rng.uniform(60, 180, n),
        "adherencia_tratamiento":       rng.uniform(0.3, 1.0, n),
        "microalbuminuria":             rng.uniform(5, 100, n),
        "ecg_anormal":                  rng.integers(0, 2, n),
        "imc":                          rng.uniform(22, 38, n),
        "edad":                         rng.integers(30, 80, n).astype(float),
        "sexo":                         rng.integers(0, 2, n),
        "hospitalizacion_previa_12m":   rng.integers(0, 2, n),
        "descompensacion_glicemica_90d": rng.integers(0, 2, n),
    })
    ruta = tmp_path / "test.csv"
    df.to_csv(ruta, index=False)
    return ruta, df

def test_generar_waterfall_shap_retorna_metadata_completa(patched, tmp_path):
    meta = generar_waterfall_shap(
        datos_paciente=DATOS_PACIENTE,
        ruta_salida=tmp_path / "waterfall.png",
    )
    for clave in (
        "paciente_id", "riesgo_predicho", "base_value", "shap_sum",
        "ruta_plot", "top_5_factores_aumentan_riesgo", "top_5_factores_reducen_riesgo",
    ):
        assert clave in meta, f"Clave faltante: {clave}"

def test_generar_waterfall_shap_crea_archivo_png(patched, tmp_path):
    ruta = tmp_path / "waterfall.png"
    generar_waterfall_shap(datos_paciente=DATOS_PACIENTE, ruta_salida=ruta)
    assert ruta.exists(), "El archivo PNG no fue creado"
    assert ruta.stat().st_size > 0, "El archivo PNG está vacío"

def test_metadata_top_factores_son_listas_de_dicts(patched, tmp_path):
    meta = generar_waterfall_shap(
        datos_paciente=DATOS_PACIENTE,
        ruta_salida=tmp_path / "waterfall.png",
    )
    for clave in ("top_5_factores_aumentan_riesgo", "top_5_factores_reducen_riesgo"):
        factores = meta[clave]
        assert isinstance(factores, list)
        for f in factores:
            assert isinstance(f, dict)
            assert {"feature", "valor", "shap_value"} <= f.keys()

def test_shap_sum_mas_base_aproxima_probabilidad(tmp_path):
    modelo = MagicMock()
    modelo.predict_proba.return_value = np.array([[0.70, 0.30]])
    pkl = {"modelo": modelo, "columnas": FEATURES_TEST}

    exp = MagicMock()
    exp.shap_values.return_value = SHAP_VALS.copy()
    exp.expected_value = BASE_VALUE

    with patch("joblib.load", return_value=pkl):
        with patch("shap.TreeExplainer", return_value=exp):
            with patch("shap.plots.waterfall", side_effect=_waterfall_stub):
                meta = generar_waterfall_shap(
                    datos_paciente=DATOS_PACIENTE,
                    ruta_salida=tmp_path / "w.png",
                )

    reconstruido = meta["base_value"] + meta["shap_sum"]
    assert abs(reconstruido - meta["riesgo_predicho"]) < 0.01, (
        f"base({meta['base_value']}) + shap_sum({meta['shap_sum']}) = {reconstruido:.4f} "
        f"≠ riesgo_predicho({meta['riesgo_predicho']})"
    )

def _shap_array_global(n):
    sv = np.zeros((n, len(_FEATS12)))
    sv[:, 0] = 0.30
    sv[:, 5] = 0.10
    return sv

def test_importancia_global_estructura_y_pct(fake_test_csv):
    ruta_csv, df = fake_test_csv
    pkl = {"modelo": MagicMock(), "columnas": list(_FEATS12)}
    exp = MagicMock()
    exp.shap_values.return_value = _shap_array_global(len(df))
    with patch("joblib.load", return_value=pkl):
        with patch("shap.TreeExplainer", return_value=exp):
            df_imp = calcular_importancia_global(ruta_test=ruta_csv)
    assert list(df_imp.columns) == ["feature", "media_abs_shap", "pct"]
    assert len(df_imp) == len(_FEATS12)
    assert df_imp["media_abs_shap"].is_monotonic_decreasing
    assert abs(df_imp["pct"].sum() - 100.0) < 1e-6

def test_importancia_global_ranking(fake_test_csv):
    ruta_csv, df = fake_test_csv
    pkl = {"modelo": MagicMock(), "columnas": list(_FEATS12)}
    exp = MagicMock()
    exp.shap_values.return_value = _shap_array_global(len(df))
    with patch("joblib.load", return_value=pkl):
        with patch("shap.TreeExplainer", return_value=exp):
            df_imp = calcular_importancia_global(ruta_test=ruta_csv)
    assert df_imp.iloc[0]["feature"] == _FEATS12[0]
    assert df_imp.iloc[1]["feature"] == _FEATS12[5]
