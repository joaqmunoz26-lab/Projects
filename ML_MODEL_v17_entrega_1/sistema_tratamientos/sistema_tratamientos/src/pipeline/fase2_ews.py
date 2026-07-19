import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decisiones import ViaRuteo
from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import CanalMonitoreo
from motor.motor_ml import FactoryModelo, ModeloRiesgoInterface
from motor.reglas_duras import MotorReglasDuras
from motor.ruteador_accion import RuteadorAccion
from servicios.hitl_revision import ColaRevisionHITL, TipoSolicitudHITL


class EWSPipeline:
    def __init__(
        self,
        modelo:       ModeloRiesgoInterface | None = None,
        ruteador:     RuteadorAccion | None        = None,
        motor_reglas: MotorReglasDuras | None      = None,
        cola_hitl:    ColaRevisionHITL | None      = None,
    ):
        self._modelo       = modelo
        self._ruteador     = ruteador
        self._motor_reglas = motor_reglas
        self._cola_hitl    = cola_hitl or ColaRevisionHITL()

    def procesar(self, ctx: PacienteContext,
                 evento: str = "control_programado",
                 dias_hasta_fin_medicamento: int | None = None) -> PacienteContext:
        self._validar_precondiciones(ctx)
        ctx.fase_actual = FaseActual.FASE_2_EWS

        resultado_duras = self._motor_reglas_inst().evaluar(ctx)
        if resultado_duras.matcheo:
            return self._ejecutar_via_roja(ctx)

        modelo = self._modelo_inst()
        riesgo = modelo.predecir(ctx.datos_clinicos)
        ctx.modelo_usado = modelo.__class__.__name__
        via, _ = self._ruteador_inst().aplicar_a_contexto(ctx, riesgo)

        if via == ViaRuteo.VERDE:
            return self._ejecutar_via_verde(ctx, dias_hasta_fin_medicamento)
        if via == ViaRuteo.AMARILLA:
            return self._ejecutar_via_amarilla(ctx)
        return self._ejecutar_via_roja(ctx)


    def _validar_precondiciones(self, ctx: PacienteContext) -> None:
        if not ctx.dm2_confirmado:
            raise ValueError(
                f"Paciente {ctx.paciente_id}: DM2 no confirmada. "
                "Ejecutar Fase 1 antes de Fase 2."
            )
        if not ctx.datos_clinicos:
            raise ValueError(
                f"Paciente {ctx.paciente_id}: datos_clinicos vacios."
            )


    def _ejecutar_via_verde(self, ctx: PacienteContext,
                            dias_hasta_fin_medicamento: int | None = None
                            ) -> PacienteContext:
        fecha_antes      = ctx.fecha_proximo_control_rutina
        adelantado_antes = ctx.control_rutina_adelantado

        self._cola_hitl.encolar(
            ctx, TipoSolicitudHITL.ORDEN_VERDE,
            dias_hasta_fin_medicamento=dias_hasta_fin_medicamento,
            especialista_titular=ctx.especialista_titular,
        )
        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ews_pipeline",
            accion="via_verde_continuar",
            justificacion="Riesgo bajo: cita rutinaria intacta, HITL ORDEN_VERDE encolado.",
            datos_usados={"riesgo_ml": ctx.riesgo_ml},
        )

        assert ctx.fecha_proximo_control_rutina == fecha_antes, \
            "REGLA ETICA VIOLADA: Via Verde modifico fecha_proximo_control_rutina"
        assert ctx.control_rutina_adelantado == adelantado_antes, \
            "REGLA ETICA VIOLADA: Via Verde modifico control_rutina_adelantado"
        return ctx

    def _ejecutar_via_amarilla(self, ctx: PacienteContext) -> PacienteContext:
        ctx.control_rutina_adelantado = True
        modalidad = self._decidir_modalidad_amarilla(ctx)

        self._cola_hitl.encolar(
            ctx, TipoSolicitudHITL.REVISION_EXAMENES,
            especialista_titular=ctx.especialista_titular,
        )
        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ews_pipeline",
            accion=f"via_amarilla_{modalidad.lower()}",
            justificacion=(
                f"Riesgo moderado: cita adelantada, modalidad={modalidad}, "
                "HITL REVISION_EXAMENES encolado."
            ),
            datos_usados={"riesgo_ml": ctx.riesgo_ml, "modalidad": modalidad},
        )
        return ctx

    def _ejecutar_via_roja(self, ctx: PacienteContext) -> PacienteContext:
        fecha_antes = ctx.fecha_proximo_control_rutina

        from pipeline.fase3_fallback import agendar_presencial_urgente

        especialista_id = ctx.especialista_titular or "doc_001"
        agendar_presencial_urgente(ctx, especialista_id)

        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ews_pipeline",
            accion="via_roja_presencial_urgente",
            justificacion="Riesgo alto/critico: cita urgente agendada, rutina intacta.",
            datos_usados={
                "riesgo_ml":        ctx.riesgo_ml,
                "barrera_activada": ctx.barrera_activada.value,
            },
        )

        assert ctx.fecha_proximo_control_rutina == fecha_antes, \
            "REGLA ETICA VIOLADA: Via Roja modifico fecha_proximo_control_rutina"
        return ctx


    def _decidir_modalidad_amarilla(self, ctx: PacienteContext) -> str:
        perfil = ctx.perfil_paciente
        if (perfil.tiene_barreras_acceso_fisico
                and perfil.canal_monitoreo == CanalMonitoreo.TENS_INGRESA):
            return "PRESENCIAL_CONTROL"
        return "TELEMEDICINA_SINCRONICA"

    def _modelo_inst(self) -> ModeloRiesgoInterface:
        if self._modelo is None:
            self._modelo = FactoryModelo.crear("xgboost")
        return self._modelo

    def _ruteador_inst(self) -> RuteadorAccion:
        if self._ruteador is None:
            self._ruteador = RuteadorAccion()
        return self._ruteador

    def _motor_reglas_inst(self) -> MotorReglasDuras:
        if self._motor_reglas is None:
            self._motor_reglas = MotorReglasDuras()
        return self._motor_reglas

if __name__ == "__main__":
    from datetime import datetime, timedelta

    from core.perfiles_tipicos import perfil_urbano_digital

    ctx = PacienteContext(
        paciente_id="DEMO_FASE2",
        perfil_paciente=perfil_urbano_digital(),
        fase_actual=FaseActual.FASE_1_INGESTA,
        dm2_confirmado=True,
        datos_clinicos={
            "hba1c": 7.8, "glucosa_ayunas": 145.0,
            "adherencia_tratamiento": 0.75,
            "presion_sistolica": 128.0, "presion_diastolica": 82.0,
            "funcion_renal_egfr": 68.0,
        },
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        especialista_titular="doc_001",
    )

    pipeline = EWSPipeline()
    try:
        resultado = pipeline.procesar(ctx)
        print(f"Via:              {resultado.via_ruteo.value if resultado.via_ruteo else 'N/A'}")
        print(f"Riesgo ML:        {resultado.riesgo_ml}")
        print(f"Barrera:          {resultado.barrera_activada.value}")
        print(f"Requiere HITL:    {resultado.requiere_hitl}")
        print(f"Esp. asignado:    {resultado.especialista_asignado}")
        print(f"Rutina adelantada:{resultado.control_rutina_adelantado}")
    except FileNotFoundError as e:
        print(f"[DEMO] Modelo no disponible: {e}")
        print("[DEMO] Ejecutar python src/06a_train_ml.py primero.")
