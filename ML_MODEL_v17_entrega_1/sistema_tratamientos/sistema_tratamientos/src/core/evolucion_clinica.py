import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importlib import import_module

_rules = import_module("02b_clinical_rules")

@dataclass
class CambioClinico:
    control_num:        int
    via_previa:         str
    hba1c_previa:       float
    hba1c_actual:       float
    delta_hba1c:        float
    adherencia_previa:  float
    adherencia_actual:  float
    transicion:         str

class ModeloEvolucionClinica:
    CAMBIO_HBA1C_VIA_VERDE            = (0.0,   0.15)
    CAMBIO_HBA1C_VIA_AMARILLA         = (-0.4,  0.30)
    CAMBIO_HBA1C_VIA_ROJA             = (-0.55, 0.35)
    CAMBIO_HBA1C_NO_RESPONDEDOR_ROJA  = (-0.2,  0.30)
    CAMBIO_HBA1C_SIN_ATENCION         = (0.3,   0.25)

    CAMBIO_ADHERENCIA_VIA_VERDE    = (0.0,   0.05)
    CAMBIO_ADHERENCIA_VIA_AMARILLA = (0.05,  0.08)
    CAMBIO_ADHERENCIA_VIA_ROJA     = (0.10,  0.10)

    PROB_RESPONDEDOR = 0.85

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.rng_np = np.random.default_rng(seed)
        self.cambios_registrados: list[CambioClinico] = []

    def evolucionar_paciente(
        self,
        datos_clinicos_previos: dict,
        via_previa: str,
        control_num: int,
        paciente_id: str = "",
    ) -> dict:

        datos_nuevos = datos_clinicos_previos.copy()
        es_respondedor = self.rng.random() < self.PROB_RESPONDEDOR

        if via_previa == "VERDE":
            mu_h, sigma_h = self.CAMBIO_HBA1C_VIA_VERDE
            mu_a, sigma_a = self.CAMBIO_ADHERENCIA_VIA_VERDE
        elif via_previa == "AMARILLA":
            if es_respondedor:
                mu_h, sigma_h = self.CAMBIO_HBA1C_VIA_AMARILLA
                mu_a, sigma_a = self.CAMBIO_ADHERENCIA_VIA_AMARILLA
            else:
                mu_h, sigma_h = (0.0, 0.25)
                mu_a, sigma_a = (0.0, 0.05)
        elif via_previa == "ROJA":
            if es_respondedor:
                mu_h, sigma_h = self.CAMBIO_HBA1C_VIA_ROJA
                mu_a, sigma_a = self.CAMBIO_ADHERENCIA_VIA_ROJA
            else:
                mu_h, sigma_h = self.CAMBIO_HBA1C_NO_RESPONDEDOR_ROJA
                mu_a, sigma_a = (0.05, 0.10)
        else:
            mu_h, sigma_h = self.CAMBIO_HBA1C_VIA_AMARILLA
            mu_a, sigma_a = self.CAMBIO_ADHERENCIA_VIA_AMARILLA

        delta_hba1c  = self.rng.gauss(mu_h, sigma_h)
        hba1c_previa = datos_clinicos_previos.get("hba1c", 7.0)
        hba1c_nueva  = max(4.5, min(13.0, hba1c_previa + delta_hba1c))
        datos_nuevos["hba1c"] = round(hba1c_nueva, 2)

        glucosa_estimada = 28.7 * hba1c_nueva - 46.7
        datos_nuevos["glucosa_ayunas"] = max(70, round(
            glucosa_estimada + self.rng.gauss(0, 15), 1
        ))

        delta_adh = self.rng.gauss(mu_a, sigma_a)
        adh_previa = datos_clinicos_previos.get("adherencia_tratamiento", 0.75)
        adh_nueva  = max(0.0, min(1.0, adh_previa + delta_adh))
        datos_nuevos["adherencia_tratamiento"] = round(adh_nueva, 3)

        if via_previa in ("AMARILLA", "ROJA") and es_respondedor:
            delta_pas = self.rng.gauss(-3, 5)
            delta_pad = self.rng.gauss(-2, 4)
        else:
            delta_pas = self.rng.gauss(0, 5)
            delta_pad = self.rng.gauss(0, 3)

        datos_nuevos["presion_sistolica"] = max(90, round(
            datos_clinicos_previos.get("presion_sistolica", 130) + delta_pas, 0
        ))
        datos_nuevos["presion_diastolica"] = max(60, round(
            datos_clinicos_previos.get("presion_diastolica", 80) + delta_pad, 0
        ))

        delta_egfr = self.rng.gauss(-2, 1) if hba1c_nueva > 9.0 else self.rng.gauss(-0.5, 0.5)
        datos_nuevos["funcion_renal_egfr"] = max(15, round(
            datos_clinicos_previos.get("funcion_renal_egfr", 80) + delta_egfr, 0
        ))

        edad = datos_clinicos_previos.get("edad", 60)
        pas_nueva = datos_nuevos["presion_sistolica"]

        if "colesterol_ldl" in datos_clinicos_previos:
            datos_nuevos["colesterol_ldl"] = round(_rules.evolucionar_ldl(
                datos_clinicos_previos["colesterol_ldl"], adh_nueva, edad,
                self.rng_np
            ), 1)
        if "microalbuminuria" in datos_clinicos_previos:
            datos_nuevos["microalbuminuria"] = round(_rules.evolucionar_microalbuminuria(
                datos_clinicos_previos["microalbuminuria"], hba1c_nueva,
                pas_nueva, adh_nueva, self.rng_np
            ), 1)
        if "ecg_anormal" in datos_clinicos_previos:
            datos_nuevos["ecg_anormal"] = _rules.evolucionar_ecg(
                datos_clinicos_previos["ecg_anormal"], edad, hba1c_nueva,
                pas_nueva,
                bool(datos_clinicos_previos.get("hospitalizacion_previa_12m", 0)),
                self.rng_np
            )

        if delta_hba1c < -0.2:
            transicion = "MEJORA"
        elif delta_hba1c > 0.2:
            transicion = "DETERIORO"
        else:
            transicion = "ESTABLE"

        self.cambios_registrados.append(CambioClinico(
            control_num=control_num,
            via_previa=via_previa,
            hba1c_previa=round(hba1c_previa, 2),
            hba1c_actual=round(hba1c_nueva, 2),
            delta_hba1c=round(delta_hba1c, 3),
            adherencia_previa=round(adh_previa, 3),
            adherencia_actual=round(adh_nueva, 3),
            transicion=transicion,
        ))

        return datos_nuevos

    def estadisticas_evolucion(self) -> dict:
        if not self.cambios_registrados:
            return {"total_cambios_registrados": 0}

        n          = len(self.cambios_registrados)
        mejoras    = sum(1 for c in self.cambios_registrados if c.transicion == "MEJORA")
        deterioros = sum(1 for c in self.cambios_registrados if c.transicion == "DETERIORO")
        estables   = n - mejoras - deterioros

        delta_prom = sum(c.delta_hba1c for c in self.cambios_registrados) / n

        return {
            "total_cambios_registrados": n,
            "mejoras":                   mejoras,
            "deterioros":                deterioros,
            "estables":                  estables,
            "tasa_mejora_pct":           round(100 * mejoras    / n, 2),
            "tasa_deterioro_pct":        round(100 * deterioros / n, 2),
            "delta_hba1c_promedio":      round(delta_prom, 3),
        }

if __name__ == "__main__":
    modelo = ModeloEvolucionClinica(seed=42)
    datos_base = {
        "hba1c": 8.5,
        "glucosa_ayunas": 200.0,
        "adherencia_tratamiento": 0.60,
        "presion_sistolica": 140,
        "presion_diastolica": 88,
        "funcion_renal_egfr": 72,
    }
    print("=== Simulación 4 controles (Roja -> Amarilla -> Verde) ===")
    datos = datos_base.copy()
    for control, via in enumerate(["ROJA", "ROJA", "AMARILLA", "VERDE"], start=2):
        datos = modelo.evolucionar_paciente(datos, via, control)
        print(f"  Control {control} (via previa {via:<8}): "
              f"HbA1c={datos['hba1c']:.2f}  adh={datos['adherencia_tratamiento']:.2f}")

    print("\n=== Estadísticas de evolución ===")
    for k, v in modelo.estadisticas_evolucion().items():
        print(f"  {k}: {v}")
