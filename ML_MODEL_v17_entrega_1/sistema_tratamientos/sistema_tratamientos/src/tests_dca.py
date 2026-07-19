import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_dca = import_module("07c_decision_curve_analysis")
calcular_beneficio_neto          = _dca.calcular_beneficio_neto
calcular_beneficio_neto_treat_all = _dca.calcular_beneficio_neto_treat_all
ejecutar_dca                      = _dca.ejecutar_dca
graficar_curvas_dca               = _dca.graficar_curvas_dca

@pytest.fixture
def datos_sinteticos():
    rng = np.random.default_rng(0)
    n   = 50
    y   = rng.integers(0, 2, n)
    p   = y.astype(float) + rng.normal(0, 0.05, n)
    p   = np.clip(p, 0.01, 0.99)
    return y, p

@pytest.fixture
def predicciones_mock(datos_sinteticos):
    y, p = datos_sinteticos
    return {"MockModelo": p}, y

def test_beneficio_neto_treat_none_es_cero(datos_sinteticos):
    y, _ = datos_sinteticos
    p_none = np.zeros(len(y))
    nb = calcular_beneficio_neto(y, p_none, umbral=0.20)
    assert nb == pytest.approx(0.0, abs=1e-9)

def test_beneficio_neto_prediccion_perfecta_positivo(datos_sinteticos):
    y, _ = datos_sinteticos
    p_perfect = y.astype(float)
    umbral = 0.20
    nb_modelo   = calcular_beneficio_neto(y, p_perfect, umbral)
    nb_treat_all = calcular_beneficio_neto_treat_all(y, umbral)
    assert nb_modelo >= nb_treat_all

def test_beneficio_neto_formula_manual():
    y      = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    proba  = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    umbral = 0.5
    nb     = calcular_beneficio_neto(y, proba, umbral)
    assert nb == pytest.approx(0.10, abs=1e-9)

def test_treat_all_decrece_con_umbral(datos_sinteticos):
    y, _ = datos_sinteticos
    nb_10 = calcular_beneficio_neto_treat_all(y, 0.10)
    nb_30 = calcular_beneficio_neto_treat_all(y, 0.30)
    nb_50 = calcular_beneficio_neto_treat_all(y, 0.50)
    assert nb_10 > nb_30 > nb_50

def test_umbral_extremo_retorna_cero():
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.1, 0.8])
    assert calcular_beneficio_neto(y, p, 0.0) == 0.0
    assert calcular_beneficio_neto(y, p, 1.0) == 0.0

def _setup_dca_test(tmp_path, n, rng_seed):
    from unittest.mock import MagicMock
    rng = np.random.default_rng(rng_seed)
    y   = rng.integers(0, 2, n)
    p   = np.clip(rng.uniform(0, 1, n), 0.01, 0.99)

    df_test = pd.DataFrame({
        "paciente_id":                  [f"P{i:04d}" for i in range(n)],
        "fecha_control":                "2024-01-01",
        "hba1c":                        rng.uniform(6.5, 12.0, n),
        "presion_sistolica":            rng.uniform(110, 180, n),
        "presion_diastolica":           rng.uniform(70, 110, n),
        "funcion_renal_egfr":           rng.uniform(40, 110, n),
        "colesterol_ldl":               rng.uniform(60, 180, n),
        "adherencia_tratamiento":       rng.uniform(0.3, 1.0, n),
        "microalbuminuria":             rng.uniform(5, 100, n),
        "ecg_anormal":                  rng.integers(0, 2, n),
        "imc":                          rng.uniform(22, 38, n),
        "edad":                         rng.integers(40, 80, n),
        "sexo":                         rng.integers(0, 2, n),
        "hospitalizacion_previa_12m":   rng.integers(0, 2, n),
        "descompensacion_glicemica_90d": y,
    })
    tmp_data = tmp_path / "data" / "processed"
    tmp_data.mkdir(parents=True)
    df_test.to_csv(tmp_data / "test.csv", index=False)

    fake_modelo = MagicMock()
    fake_modelo.predict_proba.return_value = np.column_stack([1 - p, p])
    fake_pkl = {"modelo": fake_modelo, "columnas": [
        "hba1c", "presion_sistolica", "presion_diastolica",
        "funcion_renal_egfr", "colesterol_ldl", "adherencia_tratamiento",
        "microalbuminuria", "ecg_anormal", "imc",
        "edad", "sexo", "hospitalizacion_previa_12m",
    ]}

    ruta_pkl = tmp_path / "modelo_xgboost.pkl"
    ruta_pkl.write_bytes(b"placeholder")

    return df_test, y, p, fake_pkl, ruta_pkl

def test_ejecutar_dca_retorna_dataframe_con_columnas(tmp_path, monkeypatch):
    _, y, p, fake_pkl, _ = _setup_dca_test(tmp_path, n=100, rng_seed=1)

    monkeypatch.setattr(_dca, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(_dca, "DIR_BASE",    tmp_path)
    monkeypatch.setattr(_dca.joblib, "load", lambda *a, **kw: fake_pkl)

    df_res, _, _ = ejecutar_dca(umbral_min=0.10, umbral_max=0.30, n_umbrales=5)

    assert isinstance(df_res, pd.DataFrame)
    for col in ("estrategia", "umbral", "beneficio_neto", "tipo"):
        assert col in df_res.columns, f"Columna faltante: {col}"

def test_dca_incluye_treat_all_y_treat_none(tmp_path, monkeypatch):
    _, y, p, fake_pkl, _ = _setup_dca_test(tmp_path, n=80, rng_seed=2)

    monkeypatch.setattr(_dca, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(_dca, "DIR_BASE",    tmp_path)
    monkeypatch.setattr(_dca.joblib, "load", lambda *a, **kw: fake_pkl)

    df_res, _, _ = ejecutar_dca(umbral_min=0.10, umbral_max=0.30, n_umbrales=3)

    estrategias = df_res["estrategia"].unique()
    assert "Tratar a todos"    in estrategias
    assert "No tratar a nadie" in estrategias

def test_treat_none_beneficio_neto_siempre_cero(tmp_path, monkeypatch):
    _, y, p, fake_pkl, _ = _setup_dca_test(tmp_path, n=60, rng_seed=3)

    monkeypatch.setattr(_dca, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(_dca, "DIR_BASE",    tmp_path)
    monkeypatch.setattr(_dca.joblib, "load", lambda *a, **kw: fake_pkl)

    df_res, _, _ = ejecutar_dca(umbral_min=0.10, umbral_max=0.40, n_umbrales=5)

    treat_none = df_res[df_res["estrategia"] == "No tratar a nadie"]
    assert (treat_none["beneficio_neto"] == 0.0).all()

def test_graficar_curvas_dca_crea_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    umbrales = np.linspace(0.10, 0.40, 5)
    registros = []
    for u in umbrales:
        registros.append({"estrategia": "XGBoost",        "umbral": u, "beneficio_neto": 0.05, "tipo": "modelo"})
        registros.append({"estrategia": "Tratar a todos", "umbral": u, "beneficio_neto": 0.03, "tipo": "baseline"})
        registros.append({"estrategia": "No tratar a nadie", "umbral": u, "beneficio_neto": 0.0, "tipo": "baseline"})

    df = pd.DataFrame(registros)
    ruta = tmp_path / "dca_test.png"
    graficar_curvas_dca(df, ruta)

    assert ruta.exists()
    assert ruta.stat().st_size > 1000
