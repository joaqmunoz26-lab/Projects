import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decisiones import BarreraActivada, ViaRuteo
from core.paciente_context import PacienteContext

_RAIZ = Path(__file__).resolve().parent.parent.parent

class RuteadorAccion:

    def __init__(self, ruta_yaml: str = None):
        ruta = Path(ruta_yaml) if ruta_yaml else _RAIZ / "00_clases_modalidad.yaml"
        with open(ruta, encoding="utf-8") as f:
            self.umbrales = yaml.safe_load(f)["umbrales"]

    def decidir_via(self, riesgo: float, datos_paciente: dict) -> tuple:
        u   = self.umbrales
        adh = datos_paciente.get("adherencia_tratamiento", 1.0)
        hba = datos_paciente.get("hba1c", 0.0)

        if riesgo >= u["riesgo_critico"]:
            return (ViaRuteo.ROJA,
                    f"Riesgo crítico {riesgo:.2f} >= {u['riesgo_critico']}")

        if riesgo >= u["riesgo_alto"]:
            agravante = (hba >= u["hba1c_descompensada"] or
                         adh < u["adherencia_critica"])
            if agravante:
                return (ViaRuteo.ROJA,
                        f"Riesgo alto {riesgo:.2f} con factores agravantes "
                        f"(HbA1c={hba:.1f}, adh={adh:.2f})")
            return (ViaRuteo.AMARILLA,
                    f"Riesgo alto {riesgo:.2f} sin factores agravantes")

        if riesgo >= u["riesgo_moderado"]:
            return (ViaRuteo.AMARILLA,
                    f"Riesgo moderado {riesgo:.2f} >= {u['riesgo_moderado']}")

        controlado = (adh >= u["adherencia_buena"] and
                      hba <= u["hba1c_controlada"])
        if controlado:
            return (ViaRuteo.VERDE,
                    f"Riesgo bajo {riesgo:.2f}, adherencia y HbA1c controladas")
        return (ViaRuteo.AMARILLA,
                f"Riesgo bajo {riesgo:.2f} pero adherencia baja o HbA1c no óptima "
                f"(adh={adh:.2f}, HbA1c={hba:.1f})")

    def aplicar_a_contexto(self, ctx: PacienteContext, riesgo: float) -> tuple:
        via, just = self.decidir_via(riesgo, ctx.datos_clinicos)
        ctx.riesgo_ml        = riesgo
        ctx.barrera_activada = BarreraActivada.MODELO_ML
        ctx.via_ruteo        = via
        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ruteador",
            accion=f"via_{via.value}", justificacion=just,
            datos_usados={"riesgo_ml": riesgo,
                          "adherencia": ctx.datos_clinicos.get("adherencia_tratamiento"),
                          "hba1c":      ctx.datos_clinicos.get("hba1c")},
        )
        return via, just

if __name__ == "__main__":
    r = RuteadorAccion()
    casos = [
        (0.82, {"adherencia_tratamiento": 0.90, "hba1c": 7.0},  "Riesgo crítico"),
        (0.60, {"adherencia_tratamiento": 0.40, "hba1c": 8.5},  "Alto + agravante -> ROJA"),
        (0.60, {"adherencia_tratamiento": 0.85, "hba1c": 7.2},  "Alto sin agravante -> AMARILLA"),
        (0.40, {"adherencia_tratamiento": 0.75, "hba1c": 7.5},  "Moderado -> AMARILLA"),
        (0.15, {"adherencia_tratamiento": 0.90, "hba1c": 6.8},  "Bajo controlado -> VERDE"),
        (0.15, {"adherencia_tratamiento": 0.40, "hba1c": 6.8},  "Bajo baja adh -> AMARILLA"),
    ]
    print("=== Ruteador de Acción ===")
    for riesgo, datos, desc in casos:
        via, just = r.decidir_via(riesgo, datos)
        print(f"  {desc:<35} => VIA {via.value.upper():<10} | {just}")
