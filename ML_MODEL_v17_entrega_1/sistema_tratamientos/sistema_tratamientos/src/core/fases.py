from enum import Enum


class FaseActual(Enum):
    FASE_1_INGESTA   = "fase1_ingesta"
    FASE_2_EWS       = "fase2_ews"
    FASE_3_FALLBACK  = "fase3_fallback"
    ALTA_SISTEMA     = "alta"
