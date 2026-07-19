from enum import Enum


class ViaRuteo(Enum):
    VERDE    = "verde"
    AMARILLA = "amarilla"
    ROJA     = "roja"

class BarreraActivada(Enum):
    NINGUNA      = "ninguna"
    REGLAS_DURAS = "reglas_duras"
    MODELO_ML    = "modelo_ml"

class ResultadoAccion(Enum):
    PENDIENTE = "pendiente"
    EJECUTADA = "ejecutada"
    FALLIDA   = "fallida"
    ESCALADA  = "escalada"
