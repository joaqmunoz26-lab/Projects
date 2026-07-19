import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paciente_context import PacienteContext

_red_global = None

def set_red_especialistas(red) -> None:
    global _red_global
    _red_global = red

def _get_red():
    global _red_global
    if _red_global is None:
        from servicios.red_especialistas import RedEspecialistas
        _red_global = RedEspecialistas()
    return _red_global

def con_fallback_red(fn):
    @functools.wraps(fn)
    def _wrapper(ctx: PacienteContext, especialista_id: str | None, *args, **kwargs):
        red = _get_red()

        tipo, esp_id = red.asignar_con_cascade(ctx.perfil_paciente.zona)

        ctx.pool_tipo_asignacion = tipo

        if tipo == "nativo":
            ctx.especialista_asignado    = esp_id
            ctx.especialista_respaldo_usado = False
            especialista_id              = esp_id
        elif tipo == "prestamo":
            ctx.especialista_asignado    = esp_id
            ctx.especialista_respaldo_usado = True
            especialista_id              = esp_id
        else:
            ctx.especialista_asignado = None
            especialista_id           = None

        if ctx.especialista_asignado:
            red.notificar_feedback_historial(
                ctx.especialista_asignado, ctx.paciente_id, "asignacion_caso"
            )
        return fn(ctx, especialista_id, *args, **kwargs)

    return _wrapper

@con_fallback_red
def agendar_presencial_urgente(
    ctx: PacienteContext, especialista_id: str | None
) -> None:
    ctx.log_decision(
        fase=ctx.fase_actual.value, motor="fase3_fallback",
        accion="agendar_presencial_urgente",
        justificacion=(
            f"Cita urgente con {especialista_id or 'sin_especialista'}."
        ),
        datos_usados={
            "especialista_asignado":              especialista_id,
            "requiere_intervencion_administracion": especialista_id is None,
        },
    )

if __name__ == "__main__":
    from datetime import datetime, timedelta

    from core.fases import FaseActual
    from core.perfiles_tipicos import perfil_rural_posta, perfil_urbano_digital
    from servicios.red_especialistas import RedEspecialistas

    def _ctx(pid, perfil):
        return PacienteContext(
            paciente_id=pid,
            perfil_paciente=perfil,
            fase_actual=FaseActual.FASE_2_EWS,
            dm2_confirmado=True,
            datos_clinicos={"hba1c": 8.0, "glucosa_ayunas": 160.0},
            fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
            especialista_titular="doc_001",
        )

    red = RedEspecialistas()
    set_red_especialistas(red)

    print("=== Demo Fase 3 Fallback ===\n")

    ctx1 = _ctx("P_DEMO_001", perfil_urbano_digital())
    agendar_presencial_urgente(ctx1, "endo_001")
    print(f"Caso 1 (titular ok):      asignado={ctx1.especialista_asignado}  "
          f"respaldo_usado={ctx1.especialista_respaldo_usado}")

    red.buscar_por_id("doc_001").carga_actual_consultas = 15
    ctx2 = _ctx("P_DEMO_002", perfil_urbano_digital())
    agendar_presencial_urgente(ctx2, "endo_001")
    print(f"Caso 2 (titular saturado): asignado={ctx2.especialista_asignado}  "
          f"respaldo_usado={ctx2.especialista_respaldo_usado}")

    red.buscar_por_id("doc_002").carga_actual_consultas = 15
    ctx3 = _ctx("P_DEMO_003", perfil_rural_posta())
    agendar_presencial_urgente(ctx3, "endo_001")
    print(f"Caso 3 (red saturada):     asignado={ctx3.especialista_asignado}  "
          f"respaldo_usado={ctx3.especialista_respaldo_usado}")

    set_red_especialistas(None)
