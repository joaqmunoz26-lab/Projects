import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decisiones import BarreraActivada, ViaRuteo
from core.paciente_context import PacienteContext

_ORDEN_SEVERIDAD = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2}

@dataclass
class ReglaDura:
    id:                   str
    nombre:               str
    descripcion_clinica:  str
    variables_requeridas: list
    validador:            Callable[[dict], bool]
    severidad:            str
    via_sugerida:         str
    validado_por_medico:  bool
    version:              str

    @property
    def codigo(self) -> str:
        return self.id

@dataclass
class ResultadoReglasDuras:
    matcheo:          bool
    regla_activada:   str | None  = None
    regla_nombre:     str | None  = None
    justificacion:    str            = ""
    severidad:        str | None  = None
    datos_evaluados:  dict | None = None

    @property
    def prioridad(self) -> str | None:
        return self.severidad

def _r(id, nombre, desc, vars_req, fn, sev):
    return ReglaDura(id=id, nombre=nombre, descripcion_clinica=desc,
                     variables_requeridas=vars_req, validador=fn, severidad=sev,
                     via_sugerida="ROJA", validado_por_medico=False, version="1.0")

class MotorReglasDuras:

    def __init__(self):
        self.reglas = _cargar_reglas()

    def evaluar(self, ctx: PacienteContext) -> ResultadoReglasDuras:
        datos = ctx.datos_clinicos
        for regla in sorted(self.reglas, key=lambda r: _ORDEN_SEVERIDAD.get(r.severidad, 99)):
            if not all(v in datos for v in regla.variables_requeridas):
                continue
            try:
                if not regla.validador(datos):
                    continue
            except Exception as exc:
                ctx.log_decision(
                    fase=ctx.fase_actual.value, motor="reglas_duras",
                    accion="regla_evaluacion_fallida", regla_id=regla.id,
                    justificacion=f"Evaluacion fallida: {type(exc).__name__}",
                    datos_usados={"excepcion": str(exc)[:200]},
                )
                continue
            just = f"[{regla.id}] {regla.nombre}: {regla.descripcion_clinica}"
            datos_ev = {v: datos[v] for v in regla.variables_requeridas}
            ctx.log_decision(
                fase=ctx.fase_actual.value, motor="reglas_duras",
                accion=f"bypass_ml_{regla.via_sugerida.lower()}",
                justificacion=just, datos_usados=datos_ev, regla_id=regla.id,
            )
            ctx.barrera_activada = BarreraActivada.REGLAS_DURAS
            ctx.via_ruteo        = ViaRuteo(regla.via_sugerida.lower())
            return ResultadoReglasDuras(
                matcheo=True, regla_activada=regla.id, regla_nombre=regla.nombre,
                justificacion=just, severidad=regla.severidad, datos_evaluados=datos_ev,
            )
        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="reglas_duras",
            accion="evaluacion_sin_matcheo",
            justificacion="Ninguna regla critica activada; continua a clasificacion ML",
            datos_usados={"reglas_evaluadas": [r.id for r in self.reglas]},
        )
        return ResultadoReglasDuras(matcheo=False, justificacion="Sin reglas criticas activadas.")

    @property
    def reglas_criticas(self) -> list:
        return [r for r in self.reglas if r.severidad == "CRITICA"]

def _cargar_reglas() -> list:
    return [
        _r("RD001", "Hiperglucemia crítica",
           "Glucosa > 300 mg/dL: posible cetoacidosis diabética",
           ["glucosa_ayunas"], lambda d: d.get("glucosa_ayunas", 0) > 300, "CRITICA"),
        _r("RD002", "Hipoglucemia severa",
           "Glucosa < 54 mg/dL: emergencia médica",
           ["glucosa_ayunas"], lambda d: 0 < d.get("glucosa_ayunas", 999) < 54, "CRITICA"),
        _r("RD003", "HbA1c extrema",
           "HbA1c > 11%: descompensación prolongada grave",
           ["hba1c"], lambda d: d.get("hba1c", 0) > 11, "ALTA"),
        _r("RD004", "Hipertensión maligna",
           "PAS >= 180 o PAD >= 110 mmHg: riesgo cardiovascular inmediato",
           [],
           lambda d: (d.get("presion_sistolica") is not None and d.get("presion_sistolica") >= 180)
                     or (d.get("presion_diastolica") is not None and d.get("presion_diastolica") >= 110),
           "CRITICA"),
        _r("RD005", "Función renal gravemente comprometida",
           "eGFR < 30 mL/min: requiere evaluación nefrológica urgente",
           ["funcion_renal_egfr"], lambda d: 0 < d.get("funcion_renal_egfr", 999) < 30, "ALTA"),
    ]

if __name__ == "__main__":
    from core.fases import FaseActual
    from core.perfiles import (
        CanalDiagnostico,
        CanalMonitoreo,
        PerfilPaciente,
        ZonaGeografica,
    )

    def _demo(nombre, datos):
        ctx = PacienteContext(
            paciente_id="DEMO", fase_actual=FaseActual.FASE_2_EWS,
            perfil_paciente=PerfilPaciente(
                zona=ZonaGeografica.URBANO,
                canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
                canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
            ),
            datos_clinicos=datos,
        )
        r = MotorReglasDuras().evaluar(ctx)
        estado = f"BYPASS -> {r.regla_activada} [{r.severidad}]" if r.matcheo else "sin bypass"
        print(f"  {nombre:<30} {estado}")

    print("=== Demo Barrera 1 ===")
    _demo("Glucosa 350 (RD001)",     {"glucosa_ayunas": 350})
    _demo("PAS 190 / PAD 85 (RD004)",{"presion_sistolica": 190, "presion_diastolica": 85})
    _demo("Paciente normal",          {"glucosa_ayunas": 110, "hba1c": 7.0,
                                       "presion_sistolica": 120, "presion_diastolica": 78,
                                       "funcion_renal_egfr": 72})
