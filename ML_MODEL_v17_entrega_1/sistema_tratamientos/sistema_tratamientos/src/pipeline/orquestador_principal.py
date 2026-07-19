import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import PerfilPaciente
from pipeline.fase1_ingesta import ejecutar_fase1
from pipeline.fase2_ews import EWSPipeline


def procesar_paciente(
    paciente_id: str,
    datos_clinicos: dict,
    perfil: PerfilPaciente,
    fase_inicial: FaseActual | None = None,
    evento: str = "control_programado",
    dias_hasta_fin_medicamento: int | None = None,
    fecha_proximo_control_rutina: datetime | None = None,
    especialista_titular: str | None = None,
    pipeline_ews: EWSPipeline | None = None,
    historial_controles: list | None = None,
    dm2_confirmado_inicial: bool = False,
) -> PacienteContext:
    pipeline_ews = pipeline_ews or EWSPipeline()
    fase_inicial  = fase_inicial or FaseActual.FASE_1_INGESTA

    ctx = PacienteContext(
        paciente_id=paciente_id,
        perfil_paciente=perfil,
        fase_actual=fase_inicial,
        datos_clinicos=datos_clinicos,
        historial_controles=historial_controles or [],
        fecha_proximo_control_rutina=fecha_proximo_control_rutina,
        especialista_titular=especialista_titular,
        dm2_confirmado=dm2_confirmado_inicial,
    )

    ctx.log_decision(
        fase="orquestador", motor="entry_point",
        accion=f"inicio_procesamiento_{fase_inicial.value}",
        justificacion=(
            f"Paciente {paciente_id} entra al EWS en "
            f"{fase_inicial.value}. Evento: {evento}."
        ),
        datos_usados={
            "evento":                 evento,
            "perfil_zona":            perfil.zona.value,
            "perfil_canal_diag":      perfil.canal_diagnostico.value,
            "perfil_canal_mon":       perfil.canal_monitoreo.value,
            "dm2_confirmado_inicial": dm2_confirmado_inicial,
        },
    )

    if ctx.fase_actual == FaseActual.FASE_1_INGESTA:
        ctx = ejecutar_fase1(ctx)

        if ctx.dm2_confirmado:
            ctx.log_decision(
                fase="orquestador", motor="transicion_fase",
                accion="fase1_a_fase2",
                justificacion="DM2 confirmado en Fase 1, continua a EWS.",
                datos_usados={
                    "fecha_confirmacion": str(ctx.fecha_confirmacion_dm2)
                },
            )
        else:
            ctx.log_decision(
                fase="orquestador", motor="entry_point",
                accion="fin_fase1_sin_confirmacion",
                justificacion=(
                    "DM2 no confirmado en Fase 1. "
                    "Requiere estudios adicionales. No avanza a Fase 2."
                ),
                datos_usados={"dm2_confirmado": False},
            )
            return ctx

    if ctx.fase_actual == FaseActual.FASE_2_EWS:
        if not ctx.dm2_confirmado:
            ctx.log_decision(
                fase="orquestador", motor="entry_point",
                accion="error_precondiciones_fase2",
                justificacion=(
                    "Se solicito Fase 2 pero DM2 no esta confirmado. "
                    "Aborta sin procesar."
                ),
                datos_usados={"dm2_confirmado": False},
            )
            return ctx

        ctx = pipeline_ews.procesar(
            ctx,
            evento=evento,
            dias_hasta_fin_medicamento=dias_hasta_fin_medicamento,
        )

    ctx.log_decision(
        fase="orquestador", motor="entry_point",
        accion="fin_procesamiento",
        justificacion=(
            f"Procesamiento completado. "
            f"Via final: {ctx.via_ruteo.value if ctx.via_ruteo else 'N/A'}. "
            f"Decisiones auditadas: {len(ctx.auditoria)}."
        ),
        datos_usados={
            "via_final":        ctx.via_ruteo.value if ctx.via_ruteo else None,
            "barrera_activada": ctx.barrera_activada.value,
            "uso_respaldo":     ctx.especialista_respaldo_usado,
            "requiere_hitl":    ctx.requiere_hitl,
        },
    )

    return ctx

def procesar_lote_pacientes(
    pacientes: list,
    pipeline_ews: EWSPipeline | None = None,
) -> list:
    pipeline_ews = pipeline_ews or EWSPipeline()
    return [procesar_paciente(pipeline_ews=pipeline_ews, **kw) for kw in pacientes]


def generar_reporte_lote(contextos: list) -> dict:
    total = len(contextos)
    if total == 0:
        return {"total_pacientes": 0}

    confirmados      = sum(1 for c in contextos if c.dm2_confirmado)
    requiere_hitl    = sum(1 for c in contextos if c.requiere_hitl)
    uso_respaldo     = sum(1 for c in contextos if c.especialista_respaldo_usado)
    ctrl_adelantado  = sum(1 for c in contextos if c.control_rutina_adelantado)

    def _pct(n): return round(100 * n / total, 2)

    return {
        "total_pacientes":              total,
        "dm2_confirmados":              confirmados,
        "tasa_confirmacion_pct":        _pct(confirmados),
        "fase_final_distribucion":      dict(Counter(c.fase_actual.value for c in contextos)),
        "via_ruteo_distribucion":       dict(Counter(
            c.via_ruteo.value if c.via_ruteo else "ninguna" for c in contextos
        )),
        "barrera_activada_distribucion": dict(Counter(
            c.barrera_activada.value for c in contextos
        )),
        "requiere_hitl":                requiere_hitl,
        "tasa_hitl_pct":                _pct(requiere_hitl),
        "uso_respaldo_red":             uso_respaldo,
        "tasa_respaldo_pct":            _pct(uso_respaldo),
        "controles_adelantados":        ctrl_adelantado,
        "tasa_adelanto_pct":            _pct(ctrl_adelantado),
    }

if __name__ == "__main__":
    from datetime import timedelta

    from core.perfiles_tipicos import perfil_rural_posta, perfil_urbano_digital

    casos = [
        ("Urbano_normal",
         perfil_urbano_digital(),
         {"hba1c": 7.2, "glucosa_ayunas": 135, "adherencia_tratamiento": 0.85,
          "presion_sistolica": 128, "presion_diastolica": 80, "funcion_renal_egfr": 72},
         True),
        ("Urbano_critico",
         perfil_urbano_digital(),
         {"hba1c": 8.2, "glucosa_ayunas": 350, "adherencia_tratamiento": 0.60,
          "presion_sistolica": 130, "presion_diastolica": 82, "funcion_renal_egfr": 68},
         True),
        ("Rural_posta",
         perfil_rural_posta(),
         {"hba1c": 7.8, "glucosa_ayunas": 145, "adherencia_tratamiento": 0.75,
          "presion_sistolica": 132, "presion_diastolica": 84, "funcion_renal_egfr": 65},
         True),
    ]

    print("=== Demo Orquestador Principal ===\n")
    for desc, perfil, datos, dm2_ini in casos:
        ctx = procesar_paciente(
            paciente_id=desc,
            datos_clinicos=datos,
            perfil=perfil,
            fase_inicial=FaseActual.FASE_2_EWS,
            dm2_confirmado_inicial=dm2_ini,
            fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
            especialista_titular="doc_001",
        )
        via      = ctx.via_ruteo.value if ctx.via_ruteo else "N/A"
        barrera  = ctx.barrera_activada.value
        riesgo   = f"{ctx.riesgo_ml:.4f}" if ctx.riesgo_ml is not None else "N/A (Barrera 1)"
        print(f"  {desc:<20} | via={via:<10} | barrera={barrera:<12} | "
              f"riesgo={riesgo} | HITL={ctx.requiere_hitl}")
