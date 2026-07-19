import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motor.motor_ml import (
    FactoryModelo,
    ModeloLogisticRegression,
    ModeloRandomForest,
    ModeloXGBoost,
)

_COLS = ["hba1c", "edad", "glucosa"]
_DATOS = {"hba1c": 8.5, "edad": 60.0, "glucosa": 180.0}

def _make_scaler(out=None):
    s = MagicMock()
    s.transform.return_value = out if out is not None else np.zeros((1, len(_COLS)))
    return s

def _make_modelo(proba=0.65):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - proba, proba]])
    return m

def test_modelo_logistic_predice_con_scaler():
    scaler = _make_scaler()
    pkl = {"modelo": _make_modelo(0.72), "scaler": scaler, "columnas": _COLS}

    with patch("motor.motor_ml.joblib.load", return_value=pkl):
        lr = ModeloLogisticRegression()

    riesgo = lr.predecir(_DATOS)

    scaler.transform.assert_called_once()
    assert riesgo == pytest.approx(0.72)

def test_modelo_logistic_explicar_aplica_scaler():
    scaler = _make_scaler()
    pkl = {"modelo": _make_modelo(), "scaler": scaler, "columnas": _COLS}

    mock_exp = MagicMock()
    mock_exp.shap_values.return_value = np.array([[0.10, 0.20, 0.30]])

    with patch("motor.motor_ml.joblib.load", return_value=pkl):
        with patch("motor.motor_ml.shap.LinearExplainer", return_value=mock_exp):
            lr = ModeloLogisticRegression()
            resultado = lr.explicar(_DATOS, top_k=3)

    scaler.transform.assert_called_once()
    assert len(resultado) == 3
    for entry in resultado:
        assert {"feature", "valor", "shap_contribution"} <= entry.keys()

def test_modelo_xgboost_no_necesita_scaler_y_funciona():
    modelo = _make_modelo(0.65)
    pkl = {"modelo": modelo, "columnas": _COLS}

    with patch("motor.motor_ml.joblib.load", return_value=pkl):
        xgb = ModeloXGBoost()

    assert xgb._scaler is None
    riesgo = xgb.predecir(_DATOS)
    assert riesgo == pytest.approx(0.65)
    modelo.predict_proba.assert_called_once()

def test_modelo_random_forest_no_necesita_scaler_y_funciona():
    modelo = _make_modelo(0.55)
    pkl = {"modelo": modelo, "columnas": _COLS}

    with patch("motor.motor_ml.joblib.load", return_value=pkl):
        rf = ModeloRandomForest()

    assert rf._scaler is None
    assert rf.predecir(_DATOS) == pytest.approx(0.55)

def test_factory_los_3_modelos_predicen_en_rango_0_1():
    scaler = _make_scaler()

    casos = [
        ("logistic",      {"modelo": _make_modelo(0.65), "scaler": scaler, "columnas": _COLS}),
        ("random_forest", {"modelo": _make_modelo(0.48), "columnas": _COLS}),
        ("xgboost",       {"modelo": _make_modelo(0.72), "columnas": _COLS}),
    ]
    for clave, pkl in casos:
        with patch("motor.motor_ml.joblib.load", return_value=pkl):
            m = FactoryModelo.crear(clave)
        riesgo = m.predecir(_DATOS)
        assert 0.0 <= riesgo <= 1.0, f"{clave}: {riesgo!r} fuera de [0, 1]"

def test_waterfall_xgboost_no_aplica_scaler_correcto(tmp_path):
    from importlib import import_module
    _predict = import_module("08_predict")

    modelo = _make_modelo(0.70)
    pkl = {"modelo": modelo, "columnas": _COLS}

    mock_exp = MagicMock()
    mock_exp.shap_values.return_value = np.array([[0.10, 0.20, 0.30]])
    mock_exp.expected_value = 0.20

    def _stub(*a, **kw):
        pass

    with patch("joblib.load", return_value=pkl):
        with patch("shap.TreeExplainer", return_value=mock_exp):
            with patch("shap.plots.waterfall", side_effect=_stub):
                meta = _predict.generar_waterfall_shap(
                    datos_paciente=_DATOS,
                    ruta_salida=tmp_path / "w.png",
                )

    for clave in ("paciente_id", "riesgo_predicho", "base_value", "shap_sum",
                  "top_5_factores_aumentan_riesgo", "top_5_factores_reducen_riesgo"):
        assert clave in meta, f"Clave faltante: {clave}"
    assert 0.0 <= meta["riesgo_predicho"] <= 1.0
