import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DIR_MODELOS = Path(__file__).resolve().parent.parent.parent / "models"

class ModeloRiesgoInterface(ABC):

    @abstractmethod
    def predecir(self, datos_paciente: dict) -> float: ...

    @abstractmethod
    def explicar(self, datos_paciente: dict, top_k: int = 5) -> list: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def nombre(self) -> str: ...

class _ModeloSklearn(ModeloRiesgoInterface):
    _PKL_NAME: str = ""
    _NOMBRE:   str = ""
    _VERSION:  str = ""

    def __init__(self):
        ruta = _DIR_MODELOS / self._PKL_NAME
        try:
            obj = joblib.load(ruta)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Modelo no encontrado: {ruta}\n"
                "Ejecuta primero: python src/06a_train_ml.py"
            )
        self._modelo    = obj["modelo"]
        self._columnas  = obj["columnas"]
        self._scaler    = obj.get("scaler", None)
        self._explainer = None

    def _construir_X(self, datos: dict) -> pd.DataFrame:
        faltantes = [c for c in self._columnas if c not in datos]
        if faltantes:
            logger.warning("faltan columnas en los datos: %s", faltantes)
            raise ValueError(
                f"motor_ml: faltan {len(faltantes)} features requeridas "
                f"por el modelo: {faltantes}"
            )
        return pd.DataFrame([datos])[list(self._columnas)]

    def predecir(self, datos: dict) -> float:
        X = self._construir_X(datos)
        X_input = self._scaler.transform(X) if self._scaler is not None else X
        return float(self._modelo.predict_proba(X_input)[0, 1])

    def _get_explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._modelo)
        return self._explainer

    def explicar(self, datos: dict, top_k: int = 5) -> list:
        X = self._construir_X(datos)
        X_input = self._scaler.transform(X) if self._scaler is not None else X
        sv = self._get_explainer().shap_values(X_input)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]
        df = pd.DataFrame({"feature": self._columnas,
                           "valor":   X.iloc[0].values,
                           "shap_contribution": sv})
        df["_abs"] = np.abs(sv)
        df = df.sort_values("_abs", ascending=False).head(top_k)
        return [{"feature": r["feature"],
                 "valor":   round(float(r["valor"]), 3),
                 "shap_contribution": round(float(r["shap_contribution"]), 4)}
                for _, r in df.iterrows()]

    def version(self) -> str: return self._VERSION
    def nombre(self) -> str:  return self._NOMBRE

class ModeloLogisticRegression(_ModeloSklearn):
    _PKL_NAME, _NOMBRE, _VERSION = "modelo_logistic.pkl", "LogisticRegression", "lr_v1.0.0"

    def _get_explainer(self):
        if self._explainer is None:
            bg = np.zeros((1, len(self._columnas)))
            self._explainer = shap.LinearExplainer(self._modelo, bg)
        return self._explainer

class ModeloRandomForest(_ModeloSklearn):
    _PKL_NAME, _NOMBRE, _VERSION = "modelo_forest.pkl", "RandomForest", "rf_v1.0.0"

class ModeloXGBoost(_ModeloSklearn):
    _PKL_NAME, _NOMBRE, _VERSION = "modelo_xgboost.pkl", "XGBoost", "xgboost_v1.0.0"

class FactoryModelo:

    MODELOS_DISPONIBLES = {
        "logistic":      ModeloLogisticRegression,
        "random_forest": ModeloRandomForest,
        "xgboost":       ModeloXGBoost,
    }

    @staticmethod
    def crear(nombre: str) -> ModeloRiesgoInterface:
        if nombre not in FactoryModelo.MODELOS_DISPONIBLES:
            raise ValueError(
                f"Modelo '{nombre}' no disponible. "
                f"Opciones: {list(FactoryModelo.MODELOS_DISPONIBLES)}"
            )
        return FactoryModelo.MODELOS_DISPONIBLES[nombre]()

    @staticmethod
    def listar_disponibles() -> list:
        return list(FactoryModelo.MODELOS_DISPONIBLES)

if __name__ == "__main__":
    print("benchmark")
    for nombre in FactoryModelo.listar_disponibles():
        try:
            m = FactoryModelo.crear(nombre)
            print(f"  {nombre:<15}: {m.nombre()} {m.version()} - OK")
        except FileNotFoundError as e:
            linea = str(e).splitlines()[0]
            print(f"  {nombre:<15}: no disponible: {linea}")
        except Exception as e:
            print(f"  {nombre:<15}: error: {e}")
