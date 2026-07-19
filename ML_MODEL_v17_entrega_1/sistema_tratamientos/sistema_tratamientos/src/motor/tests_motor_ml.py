import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.decisiones import ViaRuteo
from motor.motor_ml import (
    FactoryModelo,
    ModeloLogisticRegression,
    ModeloRandomForest,
    ModeloXGBoost,
)
from motor.ruteador_accion import RuteadorAccion

_COLS = [
    "hba1c", "presion_sistolica", "presion_diastolica",
    "funcion_renal_egfr", "colesterol_ldl", "adherencia_tratamiento",
    "microalbuminuria", "ecg_anormal", "imc",
    "edad", "sexo", "hospitalizacion_previa_12m",
]
_DATOS = {
    "hba1c": 8.5, "presion_sistolica": 140.0, "presion_diastolica": 88.0,
    "funcion_renal_egfr": 75.0, "colesterol_ldl": 110.0, "adherencia_tratamiento": 0.7,
    "microalbuminuria": 20.0, "ecg_anormal": 0, "imc": 29.0,
    "edad": 60.0, "sexo": 1, "hospitalizacion_previa_12m": 0,
}
_MOCK_PKL = lambda: {"modelo": MagicMock(), "columnas": _COLS}


@pytest.fixture(scope="module", autouse=True)
def _ensure_real_xgboost():
    import importlib
    for key in [k for k in list(sys.modules) if k.startswith("xgboost")]:
        if isinstance(sys.modules[key], MagicMock):
            del sys.modules[key]
    importlib.import_module("xgboost")
    yield

@patch("motor.motor_ml.joblib.load")
def test_factory_crea_logistic(mock_load):
    mock_load.return_value = _MOCK_PKL()
    m = FactoryModelo.crear("logistic")
    assert m.nombre() == "LogisticRegression"
    assert "lr" in m.version()

@patch("motor.motor_ml.joblib.load")
def test_factory_crea_random_forest(mock_load):
    mock_load.return_value = _MOCK_PKL()
    m = FactoryModelo.crear("random_forest")
    assert m.nombre() == "RandomForest"
    assert "rf" in m.version()

@patch("motor.motor_ml.joblib.load")
def test_factory_crea_xgboost(mock_load):
    mock_load.return_value = _MOCK_PKL()
    m = FactoryModelo.crear("xgboost")
    assert m.nombre() == "XGBoost"
    assert "xgboost" in m.version()

def test_factory_rechaza_modelo_no_oficial():
    with pytest.raises(ValueError, match="no disponible"):
        FactoryModelo.crear("lightgbm")

def test_factory_lista_3_modelos_oficiales():
    disponibles = FactoryModelo.listar_disponibles()
    assert set(disponibles) == {"logistic", "random_forest", "xgboost"}

@patch("motor.motor_ml.joblib.load", side_effect=FileNotFoundError("modelo_xgboost.pkl"))
def test_factory_modelo_faltante_da_mensaje_util(mock_load):
    with pytest.raises(FileNotFoundError, match="06a_train_ml.py"):
        FactoryModelo.crear("xgboost")

@patch("motor.motor_ml.joblib.load")
def test_xgboost_predice_en_rango_0_1(mock_load):
    mock_m = MagicMock()
    mock_m.predict_proba.return_value = np.array([[0.35, 0.65]])
    mock_load.return_value = {"modelo": mock_m, "columnas": _COLS}
    r = ModeloXGBoost().predecir(_DATOS)
    assert 0.0 <= r <= 1.0
    assert r == pytest.approx(0.65)

@patch("motor.motor_ml.shap.TreeExplainer")
@patch("motor.motor_ml.joblib.load")
def test_xgboost_explica_retorna_5_features(mock_load, mock_exp_cls):
    mock_load.return_value = {"modelo": MagicMock(), "columnas": _COLS}
    mock_exp = MagicMock()
    mock_exp.shap_values.return_value = np.array([[
        0.10, -0.20, 0.30, -0.05, 0.95, 0.02,
        0.01, 0.03, 0.06, 0.07, 0.08, 0.09,
    ]])
    mock_exp_cls.return_value = mock_exp

    exp = ModeloXGBoost().explicar(_DATOS, top_k=5)

    assert len(exp) == 5
    for e in exp:
        assert "feature" in e and "valor" in e and "shap_contribution" in e
    assert exp[0]["feature"] == "colesterol_ldl"

def test_los_3_modelos_implementan_interfaz():
    for cls in [ModeloLogisticRegression, ModeloRandomForest, ModeloXGBoost]:
        with patch.object(cls, "__init__", return_value=None):
            m = cls()
        assert isinstance(m.nombre(), str) and m.nombre()
        assert isinstance(m.version(), str) and m.version()

def test_ruteador_riesgo_alto_da_via_roja():
    via, _ = RuteadorAccion().decidir_via(
        0.80, {"adherencia_tratamiento": 0.90, "hba1c": 7.0})
    assert via == ViaRuteo.ROJA

def test_ruteador_riesgo_bajo_adherencia_alta_da_verde():
    via, _ = RuteadorAccion().decidir_via(
        0.15, {"adherencia_tratamiento": 0.90, "hba1c": 6.5})
    assert via == ViaRuteo.VERDE

def test_ruteador_riesgo_bajo_adherencia_baja_da_amarilla():
    via, _ = RuteadorAccion().decidir_via(
        0.15, {"adherencia_tratamiento": 0.40, "hba1c": 6.8})
    assert via == ViaRuteo.AMARILLA

def test_ruteador_riesgo_moderado_da_amarilla():
    via, _ = RuteadorAccion().decidir_via(
        0.40, {"adherencia_tratamiento": 0.80, "hba1c": 7.5})
    assert via == ViaRuteo.AMARILLA


def test_smoke_pkl_real_pacientes_ancla():
    import importlib

    import joblib
    import pandas as pd

    _raiz    = Path(__file__).resolve().parent.parent.parent
    pkl      = _raiz / "models" / "modelo_xgboost.pkl"
    csv_test = _raiz / "data" / "processed" / "test.csv"

    if not pkl.exists():
        pytest.skip("models/modelo_xgboost.pkl no disponible")
    if not csv_test.exists():
        pytest.skip("data/processed/test.csv no disponible")

    obj      = joblib.load(pkl)
    modelo   = obj["modelo"]
    columnas = obj["columnas"]

    sp   = importlib.import_module("05_split")
    df   = pd.read_csv(csv_test)
    X, _ = sp.separar_x_y(df)
    X    = X.reindex(columns=columnas, fill_value=0)

    anclas = {
        "P04808": 0.8766,
        "P04259": 0.1413,
        "P02499": 0.8537,
    }
    TOL = 0.001

    for pid, esperado in anclas.items():
        mascara = (df["paciente_id"] == pid) & (df["control_num"] == 8)
        assert mascara.sum() == 1, f"Fila control_num=8 no encontrada para {pid}"
        idx   = df[mascara].index[0]
        prob  = float(modelo.predict_proba(X.loc[[idx]])[0, 1])
        diff  = abs(prob - esperado)
        assert diff <= TOL, (
            f"{pid}: esperado {esperado}, obtenido {prob:.4f}, diff={diff:.4f} > {TOL}"
        )
