import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decisiones import BarreraActivada, ViaRuteo
from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from core.perfiles_tipicos import perfil_rural_posta
from motor.motor_ml import ModeloRiesgoInterface
from motor.reglas_duras import MotorReglasDuras
from motor.ruteador_accion import RuteadorAccion
from pipeline.fase2_ews import EWSPipeline
from pipeline.fase3_fallback import agendar_presencial_urgente, set_red_especialistas
from servicios.hitl_revision import ColaRevisionHITL, TipoSolicitudHITL
from servicios.red_especialistas import RedEspecialistas


class _ModeloMock(ModeloRiesgoInterface):
    def __init__(self, riesgo: float):
        self._riesgo = riesgo

    def predecir(self, datos: dict) -> float:
        return self._riesgo

    def explicar(self, datos: dict, top_k: int = 5) -> list:
        return []

    def version(self) -> str:
        return "mock_v0"

    def nombre(self) -> str:
        return "mock"

@pytest.fixture(autouse=True)
def _reset_red():
    set_red_especialistas(None)
    yield
    set_red_especialistas(None)

def _crear_ctx(
    zona: ZonaGeografica = ZonaGeografica.URBANO,
    canal_mon: CanalMonitoreo = CanalMonitoreo.APP_AUTONOMA,
    dm2: bool = True,
    datos_extra: dict = None,
) -> PacienteContext:
    ctx = PacienteContext(
        paciente_id="P_TEST_001",
        perfil_paciente=PerfilPaciente(
            zona,
            CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_mon,
        ),
        fase_actual=FaseActual.FASE_1_INGESTA,
        dm2_confirmado=dm2,
    )
    ctx.fecha_proximo_control_rutina = datetime.now() + timedelta(days=90)
    ctx.datos_clinicos = {
        "hba1c": 7.0,
        "glucosa_ayunas": 130.0,
        "adherencia_tratamiento": 0.85,
        "presion_sistolica": 125.0,
        "presion_diastolica": 78.0,
        "funcion_renal_egfr": 72.0,
    }
    if datos_extra:
        ctx.datos_clinicos.update(datos_extra)
    ctx.especialista_titular = "doc_001"
    return ctx

def _pipeline(riesgo: float, cola: ColaRevisionHITL = None) -> EWSPipeline:
    return EWSPipeline(
        modelo=_ModeloMock(riesgo),
        ruteador=RuteadorAccion(),
        motor_reglas=MotorReglasDuras(),
        cola_hitl=cola or ColaRevisionHITL(),
    )

def test_precondicion_dm2_no_confirmada_lanza_error():
    ctx = _crear_ctx(dm2=False)
    with pytest.raises(ValueError, match="DM2 no confirmada"):
        _pipeline(0.10).procesar(ctx)

def test_precondicion_datos_clinicos_vacios_lanza_error():
    ctx = _crear_ctx()
    ctx.datos_clinicos = {}
    with pytest.raises(ValueError, match="datos_clinicos vacios"):
        _pipeline(0.10).procesar(ctx)

def test_barrera1_glucosa_critica_bypass_ml_via_roja():
    set_red_especialistas(RedEspecialistas())
    ctx = _crear_ctx(datos_extra={"glucosa_ayunas": 350.0})

    _pipeline(0.05).procesar(ctx)

    assert ctx.barrera_activada == BarreraActivada.REGLAS_DURAS
    assert ctx.via_ruteo        == ViaRuteo.ROJA

def test_barrera2_riesgo_bajo_via_verde():
    ctx = _crear_ctx()
    _pipeline(0.10).procesar(ctx)

    assert ctx.via_ruteo        == ViaRuteo.VERDE
    assert ctx.barrera_activada == BarreraActivada.MODELO_ML
    assert ctx.riesgo_ml        == pytest.approx(0.10)

def test_barrera2_riesgo_moderado_via_amarilla():
    ctx = _crear_ctx()
    _pipeline(0.40).procesar(ctx)

    assert ctx.via_ruteo == ViaRuteo.AMARILLA

def test_barrera2_riesgo_critico_via_roja():
    set_red_especialistas(RedEspecialistas())
    ctx = _crear_ctx()
    _pipeline(0.80).procesar(ctx)

    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_via_verde_NO_modifica_fecha_rutinaria():
    ctx = _crear_ctx()
    fecha_antes = ctx.fecha_proximo_control_rutina

    _pipeline(0.10).procesar(ctx)

    assert ctx.via_ruteo == ViaRuteo.VERDE
    assert ctx.fecha_proximo_control_rutina == fecha_antes, \
        "Via Verde modifico fecha_proximo_control_rutina (regla etica violada)"
    assert ctx.control_rutina_adelantado is False, \
        "Via Verde marco control_rutina_adelantado=True (regla etica violada)"

def test_via_roja_NO_modifica_fecha_rutinaria():
    set_red_especialistas(RedEspecialistas())
    ctx = _crear_ctx(datos_extra={"glucosa_ayunas": 350.0})
    fecha_antes = ctx.fecha_proximo_control_rutina

    _pipeline(0.05).procesar(ctx)

    assert ctx.via_ruteo == ViaRuteo.ROJA
    assert ctx.fecha_proximo_control_rutina == fecha_antes, \
        "Via Roja modifico fecha_proximo_control_rutina (regla etica violada)"


def test_via_amarilla_adelanta_control_rutinario():
    ctx = _crear_ctx()
    _pipeline(0.40).procesar(ctx)

    assert ctx.via_ruteo              == ViaRuteo.AMARILLA
    assert ctx.control_rutina_adelantado is True

def test_via_amarilla_rural_posta_decide_presencial():
    ctx = PacienteContext(
        paciente_id="P_RURAL_POSTA",
        perfil_paciente=perfil_rural_posta(),
        fase_actual=FaseActual.FASE_1_INGESTA,
        dm2_confirmado=True,
        datos_clinicos={
            "hba1c": 7.0, "glucosa_ayunas": 130.0,
            "adherencia_tratamiento": 0.85,
        },
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        especialista_titular="doc_001",
    )
    cola = ColaRevisionHITL()
    _pipeline(0.40, cola=cola).procesar(ctx)

    assert ctx.via_ruteo == ViaRuteo.AMARILLA
    acciones = [e.accion for e in ctx.auditoria]
    assert any("presencial_control" in a for a in acciones)

def test_via_verde_encola_hitl_orden_verde():
    cola = ColaRevisionHITL()
    ctx  = _crear_ctx()
    _pipeline(0.10, cola=cola).procesar(ctx)

    assert ctx.requiere_hitl
    pendientes = cola.obtener_pendientes()
    assert len(pendientes) == 1
    assert pendientes[0].tipo == TipoSolicitudHITL.ORDEN_VERDE

def test_via_amarilla_encola_hitl_revision_examenes():
    cola = ColaRevisionHITL()
    ctx  = _crear_ctx()
    _pipeline(0.40, cola=cola).procesar(ctx)

    assert ctx.requiere_hitl
    pendientes = cola.obtener_pendientes()
    assert len(pendientes) == 1
    assert pendientes[0].tipo == TipoSolicitudHITL.REVISION_EXAMENES

def test_fase3_titular_disponible():
    red = RedEspecialistas()
    set_red_especialistas(red)
    ctx = _crear_ctx()
    ctx.fase_actual = FaseActual.FASE_2_EWS

    agendar_presencial_urgente(ctx, "doc_001")

    assert ctx.especialista_asignado       == "doc_001"
    assert ctx.especialista_respaldo_usado is False


def test_fase3_cascade_asigna_nativo_cuando_titular_saturado():
    red = RedEspecialistas()
    red.buscar_por_id("doc_001").carga_actual_consultas = 1
    set_red_especialistas(red)

    ctx = _crear_ctx(zona=ZonaGeografica.URBANO)
    ctx.fase_actual = FaseActual.FASE_2_EWS

    agendar_presencial_urgente(ctx, "doc_001")

    assert ctx.especialista_asignado is not None
    assert ctx.especialista_asignado != "doc_001"
    assert ctx.especialista_respaldo_usado is False
    assert getattr(ctx, "pool_tipo_asignacion", None) == "nativo"


def test_fase3_red_saturada_asigna_none():
    red = RedEspecialistas()
    red.buscar_por_id("doc_010").carga_actual_consultas = 1
    set_red_especialistas(red)

    ctx = _crear_ctx(zona=ZonaGeografica.RURAL_AISLADO)
    ctx.fase_actual = FaseActual.FASE_2_EWS

    agendar_presencial_urgente(ctx, "doc_010")

    assert ctx.especialista_asignado is None
    assert getattr(ctx, "pool_tipo_asignacion", None) == "saturado"


def test_via_roja_con_titular_ocupado_usa_cascade_nativo():
    red = RedEspecialistas()
    red.buscar_por_id("doc_001").carga_actual_consultas = 1
    set_red_especialistas(red)

    ctx = _crear_ctx(zona=ZonaGeografica.URBANO,
                     datos_extra={"glucosa_ayunas": 350.0})
    ctx.especialista_titular = "doc_001"

    _pipeline(0.05).procesar(ctx)

    assert ctx.via_ruteo              == ViaRuteo.ROJA
    assert ctx.barrera_activada       == BarreraActivada.REGLAS_DURAS
    assert ctx.especialista_asignado  is not None
    assert ctx.especialista_respaldo_usado is False
    assert getattr(ctx, "pool_tipo_asignacion", None) == "nativo"

class _ModeloEspia(ModeloRiesgoInterface):
    def __init__(self):
        self.predecir_llamado = False

    def predecir(self, datos: dict) -> float:
        self.predecir_llamado = True
        return 0.05

    def explicar(self, datos: dict, top_k: int = 5) -> list:
        return []

    def version(self) -> str:
        return "espia_v0"

    def nombre(self) -> str:
        return "espia"

@pytest.mark.parametrize("datos_clinicos,regla_esperada", [
    ({"glucosa_ayunas": 50},                                 "RD002"),
    ({"hba1c": 12.0},                                        "RD003"),
    ({"presion_sistolica": 190, "presion_diastolica": 85},   "RD004"),
    ({"funcion_renal_egfr": 25},                             "RD005"),
])
def test_reglas_duras_RD002_a_RD005_bypasean_modelo_ml(datos_clinicos, regla_esperada):
    set_red_especialistas(RedEspecialistas())
    espia = _ModeloEspia()
    ctx   = _crear_ctx(datos_extra=datos_clinicos)
    EWSPipeline(
        modelo=espia,
        ruteador=RuteadorAccion(),
        motor_reglas=MotorReglasDuras(),
        cola_hitl=ColaRevisionHITL(),
    ).procesar(ctx)

    assert not espia.predecir_llamado, \
        f"{regla_esperada}: predecir() invocado a pesar de regla dura activa"
    assert ctx.barrera_activada == BarreraActivada.REGLAS_DURAS, \
        f"{regla_esperada}: barrera_activada no es REGLAS_DURAS"
    assert ctx.via_ruteo == ViaRuteo.ROJA, \
        f"{regla_esperada}: via_ruteo no es ROJA"

def test_via_roja_ml_path_NO_modifica_fecha_rutinaria():
    set_red_especialistas(RedEspecialistas())
    ctx         = _crear_ctx()
    fecha_antes = ctx.fecha_proximo_control_rutina

    _pipeline(0.80).procesar(ctx)

    assert ctx.via_ruteo        == ViaRuteo.ROJA
    assert ctx.barrera_activada == BarreraActivada.MODELO_ML
    assert ctx.fecha_proximo_control_rutina == fecha_antes, \
        "Via Roja (path ML) modifico fecha_proximo_control_rutina (regla etica violada)"
