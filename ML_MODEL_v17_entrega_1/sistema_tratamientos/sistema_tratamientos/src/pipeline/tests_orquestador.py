import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decisiones import BarreraActivada, ViaRuteo
from core.fases import FaseActual
from core.perfiles_tipicos import perfil_urbano_digital
from motor.motor_ml import ModeloRiesgoInterface
from motor.reglas_duras import MotorReglasDuras
from motor.ruteador_accion import RuteadorAccion
from pipeline.fase2_ews import EWSPipeline
from pipeline.fase3_fallback import set_red_especialistas
from pipeline.orquestador_principal import (
    generar_reporte_lote,
    procesar_lote_pacientes,
    procesar_paciente,
)
from servicios.hitl_revision import ColaRevisionHITL
from servicios.red_especialistas import RedEspecialistas


class _ModeloMock(ModeloRiesgoInterface):
    def __init__(self, riesgo: float = 0.10):
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

def _pipeline(riesgo: float = 0.10) -> EWSPipeline:
    return EWSPipeline(
        modelo=_ModeloMock(riesgo),
        ruteador=RuteadorAccion(),
        motor_reglas=MotorReglasDuras(),
        cola_hitl=ColaRevisionHITL(),
    )

def _datos_dm2_confirmados() -> dict:
    return {
        "hba1c": 7.0,
        "glucosa_ayunas": 135.0,
        "adherencia_tratamiento": 0.85,
        "presion_sistolica": 125.0,
        "presion_diastolica": 78.0,
        "funcion_renal_egfr": 72.0,
    }

def _datos_dm2_NO_confirmados() -> dict:
    return {
        "hba1c": 5.8,
        "glucosa_ayunas": 90.0,
        "adherencia_tratamiento": 0.85,
        "presion_sistolica": 120.0,
        "presion_diastolica": 75.0,
        "funcion_renal_egfr": 80.0,
    }

def test_orquestador_entry_default_es_fase1():
    ctx = procesar_paciente(
        paciente_id="T_DEFAULT",
        datos_clinicos=_datos_dm2_confirmados(),
        perfil=perfil_urbano_digital(),
        pipeline_ews=_pipeline(0.10),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
    )
    fases_auditadas = {d.fase for d in ctx.auditoria}
    assert "fase1_ingesta" in fases_auditadas


def test_orquestador_paciente_directo_fase2_si_dm2_confirmado_inicial():
    ctx = procesar_paciente(
        paciente_id="T_FASE2_DIRECTO",
        datos_clinicos=_datos_dm2_confirmados(),
        perfil=perfil_urbano_digital(),
        fase_inicial=FaseActual.FASE_2_EWS,
        dm2_confirmado_inicial=True,
        pipeline_ews=_pipeline(0.10),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
    )
    fases_auditadas = {d.fase for d in ctx.auditoria}
    assert "fase1_ingesta" not in fases_auditadas
    assert ctx.via_ruteo is not None

def test_orquestador_fase1_sin_confirmacion_no_avanza():
    ctx = procesar_paciente(
        paciente_id="T_SIN_DM2",
        datos_clinicos=_datos_dm2_NO_confirmados(),
        perfil=perfil_urbano_digital(),
    )
    assert ctx.dm2_confirmado is False
    assert ctx.via_ruteo is None
    fases_auditadas = {d.fase for d in ctx.auditoria}
    assert "fase2_ews" not in fases_auditadas

def test_orquestador_fase2_sin_dm2_confirmado_aborta():
    ctx = procesar_paciente(
        paciente_id="T_ABORTA",
        datos_clinicos=_datos_dm2_confirmados(),
        perfil=perfil_urbano_digital(),
        fase_inicial=FaseActual.FASE_2_EWS,
        dm2_confirmado_inicial=False,
        pipeline_ews=_pipeline(),
    )
    assert ctx.via_ruteo is None
    acciones = [d.accion for d in ctx.auditoria]
    assert any("error_precondiciones" in a for a in acciones)

def test_orquestador_glucosa_critica_bypassea_ml():
    set_red_especialistas(RedEspecialistas())
    datos = _datos_dm2_confirmados()
    datos["glucosa_ayunas"] = 350.0

    ctx = procesar_paciente(
        paciente_id="T_GLUCOSA_CRIT",
        datos_clinicos=datos,
        perfil=perfil_urbano_digital(),
        fase_inicial=FaseActual.FASE_2_EWS,
        dm2_confirmado_inicial=True,
        pipeline_ews=_pipeline(riesgo=0.05),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        especialista_titular="doc_001",
    )
    assert ctx.via_ruteo        == ViaRuteo.ROJA
    assert ctx.barrera_activada == BarreraActivada.REGLAS_DURAS

def test_orquestador_via_roja_con_red_saturada_marca_alerta():
    red = RedEspecialistas()
    for doc_id in [f"doc_{i:03d}" for i in range(1, 7)]:
        red.buscar_por_id(doc_id).carga_actual_consultas = 15
    set_red_especialistas(red)

    datos = _datos_dm2_confirmados()
    datos["glucosa_ayunas"] = 350.0

    ctx = procesar_paciente(
        paciente_id="T_RED_SAT",
        datos_clinicos=datos,
        perfil=perfil_urbano_digital(),
        fase_inicial=FaseActual.FASE_2_EWS,
        dm2_confirmado_inicial=True,
        pipeline_ews=_pipeline(riesgo=0.05),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        especialista_titular="doc_001",
    )
    assert ctx.via_ruteo             == ViaRuteo.ROJA
    assert ctx.especialista_asignado is None

def test_procesar_lote_5_pacientes_genera_5_resultados():
    pipeline = _pipeline(0.10)
    pacientes = [
        dict(
            paciente_id=f"LOTE_{i:02d}",
            datos_clinicos=_datos_dm2_confirmados(),
            perfil=perfil_urbano_digital(),
            fase_inicial=FaseActual.FASE_2_EWS,
            dm2_confirmado_inicial=True,
            fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        )
        for i in range(5)
    ]
    resultados = procesar_lote_pacientes(pacientes, pipeline_ews=pipeline)
    assert len(resultados) == 5
    assert all(r.via_ruteo is not None for r in resultados)

def test_reporte_lote_calcula_estadisticas():
    pipeline = _pipeline(0.10)
    pacientes = [
        dict(
            paciente_id=f"REP_{i}",
            datos_clinicos=_datos_dm2_confirmados(),
            perfil=perfil_urbano_digital(),
            fase_inicial=FaseActual.FASE_2_EWS,
            dm2_confirmado_inicial=True,
            fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        )
        for i in range(4)
    ]
    contextos = procesar_lote_pacientes(pacientes, pipeline_ews=pipeline)
    reporte   = generar_reporte_lote(contextos)

    assert reporte["total_pacientes"]       == 4
    assert reporte["tasa_confirmacion_pct"] == 100.0
    assert "via_ruteo_distribucion"         in reporte
    assert "barrera_activada_distribucion"  in reporte
    assert "tasa_hitl_pct"                  in reporte

def test_reporte_lote_vacio_no_crashea():
    reporte = generar_reporte_lote([])
    assert reporte == {"total_pacientes": 0}

def test_orquestador_registra_3_capas_de_decisiones():
    ctx = procesar_paciente(
        paciente_id="T_AUDIT",
        datos_clinicos=_datos_dm2_confirmados(),
        perfil=perfil_urbano_digital(),
        pipeline_ews=_pipeline(0.10),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
        especialista_titular="doc_001",
    )
    fases_auditadas = {d.fase for d in ctx.auditoria}
    assert "orquestador"   in fases_auditadas
    assert "fase1_ingesta" in fases_auditadas
    assert "fase2_ews"     in fases_auditadas

def test_snapshot_serializable_post_orquestacion():
    ctx = procesar_paciente(
        paciente_id="T_SNAP",
        datos_clinicos=_datos_dm2_confirmados(),
        perfil=perfil_urbano_digital(),
        fase_inicial=FaseActual.FASE_2_EWS,
        dm2_confirmado_inicial=True,
        pipeline_ews=_pipeline(0.10),
        fecha_proximo_control_rutina=datetime.now() + timedelta(days=90),
    )
    snap         = ctx.snapshot()
    serializado  = json.dumps(snap)

    assert isinstance(serializado, str)
    assert snap["paciente_id"] == "T_SNAP"
    assert snap["via_ruteo"]   is not None
