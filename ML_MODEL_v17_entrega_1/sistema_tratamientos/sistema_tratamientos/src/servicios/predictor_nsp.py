from dataclasses import dataclass

import numpy as np

COSTO_WHATSAPP_CLP = 35
COSTO_SMS_FALLBACK_CLP = 75
COSTO_LLAMADA_PERSONALIZADA_CLP = 700

REDUCCION_ESCENARIO_A_DOBLE = 0.52
REDUCCION_ESCENARIO_B_LLAMADA = 0.49
REDUCCION_ESCENARIO_C_DIGITAL = 0.27

PROB_WHATSAPP_URBANO          = 0.90
PROB_WHATSAPP_RURAL_CERCANO   = 0.80
PROB_WHATSAPP_RURAL_AISLADO   = 0.65

UMBRAL_RIESGO_BAJO  = 25
UMBRAL_RIESGO_MEDIO = 50
UMBRAL_RIESGO_ALTO  = 75

@dataclass
class ScoreNSP:
    paciente_id: str
    score_total: int
    categoria: str
    desglose_componentes: dict

class PredictorNSP:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def _prob_whatsapp(self, zona: str) -> float:
        zona_upper = zona.upper()
        if "RURAL_AISLADO" in zona_upper:
            return PROB_WHATSAPP_RURAL_AISLADO
        if "RURAL" in zona_upper:
            return PROB_WHATSAPP_RURAL_CERCANO
        return PROB_WHATSAPP_URBANO

    def calcular_score(self, paciente_estado: dict) -> ScoreNSP:
        desglose: dict = {}
        score = 0

        h = int(paciente_estado.get("historial_inasistencias", 0))
        p1 = 30 if h >= 3 else (20 if h == 2 else (10 if h == 1 else 0))
        score += p1
        desglose["historial_inasistencias"] = p1

        via = str(paciente_estado.get("via_clinica", "AMARILLA")).upper()
        if via in ("ROJA",):
            p2 = -10
        elif via in ("VERDE",):
            p2 = 5
        else:
            p2 = 0
        score += p2
        desglose["via_clinica"] = p2

        p3 = 15 if str(paciente_estado.get("tramo_fonasa", "B")).upper() == "A" else 0
        score += p3
        desglose["tramo_fonasa"] = p3

        hba1c = float(paciente_estado.get("hba1c_actual", 7.5))
        p4 = 10 if hba1c < 7.0 else (5 if hba1c < 7.5 else 0)
        score += p4
        desglose["hba1c_controlado"] = p4

        modalidad = str(paciente_estado.get("modalidad", "")).lower()
        zona = str(paciente_estado.get("zona_geografica", "URBANO")).upper()
        if "presencial" in modalidad and "RURAL" in zona:
            p5 = 10
        elif "presencial" in modalidad:
            p5 = 5
        else:
            p5 = 0
        score += p5
        desglose["barrera_fisica"] = p5

        edad = int(paciente_estado.get("edad", 50))
        p6 = 5 if edad >= 70 else (3 if edad >= 60 else 0)
        score += p6
        desglose["edad"] = p6

        p7 = 5 if not paciente_estado.get("acceso_internet", True) else 0
        score += p7
        desglose["sin_internet"] = p7

        dias = int(paciente_estado.get("dias_sin_registro", 0))
        p8 = 15 if dias > 30 else (10 if dias > 14 else (5 if dias > 7 else 0))
        score += p8
        desglose["inercia_paciente"] = p8

        score = max(0, min(100, score))

        if score >= UMBRAL_RIESGO_ALTO:
            categoria = "CRITICO"
        elif score >= UMBRAL_RIESGO_MEDIO:
            categoria = "ALTO"
        elif score >= UMBRAL_RIESGO_BAJO:
            categoria = "MEDIO"
        else:
            categoria = "BAJO"

        return ScoreNSP(
            paciente_id=str(paciente_estado.get("paciente_id", "")),
            score_total=score,
            categoria=categoria,
            desglose_componentes=desglose,
        )

    def decidir_intervencion(
        self,
        score_nsp: ScoreNSP,
        zona: str,
        via_clinica: str = "AMARILLA",
    ) -> tuple:
        if score_nsp.categoria == "CRITICO":
            tiene_wa = self._rng.random() < self._prob_whatsapp(zona)
            canal_dig = "whatsapp" if tiene_wa else "sms"
            costo_dig = COSTO_WHATSAPP_CLP if tiene_wa else COSTO_SMS_FALLBACK_CLP
            return (
                "A_doble",
                f"{canal_dig}+llamada",
                costo_dig + COSTO_LLAMADA_PERSONALIZADA_CLP,
                REDUCCION_ESCENARIO_A_DOBLE,
            )

        if score_nsp.categoria == "ALTO":
            _P_BASE_VIA = {"VERDE": 0.20, "AMARILLA": 0.10, "ROJA": 0.05}
            p_base = _P_BASE_VIA.get(via_clinica.upper(), 0.10)
            if p_base >= 0.138:
                return (
                    "B_llamada",
                    "llamada",
                    COSTO_LLAMADA_PERSONALIZADA_CLP,
                    REDUCCION_ESCENARIO_B_LLAMADA,
                )
            tiene_wa = self._rng.random() < self._prob_whatsapp(zona)
            canal = "whatsapp" if tiene_wa else "sms"
            costo = COSTO_WHATSAPP_CLP if tiene_wa else COSTO_SMS_FALLBACK_CLP
            return ("C_digital", canal, costo, REDUCCION_ESCENARIO_C_DIGITAL)

        if score_nsp.categoria == "MEDIO":
            tiene_wa = self._rng.random() < self._prob_whatsapp(zona)
            canal = "whatsapp" if tiene_wa else "sms"
            costo = COSTO_WHATSAPP_CLP if tiene_wa else COSTO_SMS_FALLBACK_CLP
            return ("C_digital", canal, costo, REDUCCION_ESCENARIO_C_DIGITAL)

        return ("ninguna", "ninguno", 0, 0.0)
