import json
import sys
from pathlib import Path

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


def _ctx_rural_digital() -> PacienteContext:
    return PacienteContext(
        paciente_id="TEST001",
        perfil_paciente=PerfilPaciente(
            zona=ZonaGeografica.RURAL_AISLADO,
            canal_diagnostico=CanalDiagnostico.TELECONSULTA_DOMICILIO,
            canal_monitoreo=CanalMonitoreo.WHATSAPP_AUTONOMO,
        ),
        fase_actual=FaseActual.FASE_1_INGESTA,
    )

def _ctx_urbano_digital() -> PacienteContext:
    return PacienteContext(
        paciente_id="TEST002",
        perfil_paciente=PerfilPaciente(
            zona=ZonaGeografica.URBANO,
            canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_1_INGESTA,
    )

def test_perfil_rural_aislado_teleconsulta_domicilio():
    p = _ctx_rural_digital().perfil_paciente
    assert p.tiene_barreras_acceso_fisico is True
    assert p.es_autonomo_digital is True
    assert p.requiere_tens_facilitador is False
    assert p.requiere_tens_ingresador is False

def test_perfil_urbano_presencial_no_tiene_barreras():
    p = _ctx_urbano_digital().perfil_paciente
    assert p.tiene_barreras_acceso_fisico is False
    assert p.es_autonomo_digital is True
    assert p.requiere_tens_facilitador is False
    assert p.requiere_tens_ingresador is False

def test_perfil_rural_posta_requiere_tens_en_ambas_fases():
    p = PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
    )
    assert p.requiere_tens_facilitador is True
    assert p.requiere_tens_ingresador is True
    assert p.es_autonomo_digital is False

def test_perfil_urbano_mayor_tens_ingresador_no_tiene_barreras():
    p = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
    )
    assert p.tiene_barreras_acceso_fisico is False
    assert p.requiere_tens_facilitador is False
    assert p.requiere_tens_ingresador is True
    assert p.es_autonomo_digital is False

def test_perfil_descripcion_legible():
    p = PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
    )
    desc = p.descripcion_legible
    assert "Rural aislado" in desc
    assert "TENS facilita" in desc
    assert "TENS ingresa" in desc

def test_contexto_log_decision_agrega_a_auditoria():
    ctx = _ctx_rural_digital()
    assert len(ctx.auditoria) == 0
    ctx.log_decision(
        fase="fase1_ingesta", motor="reglas_duras", accion="confirmar_dm2",
        justificacion="HbA1c 7.2 >= 6.5", datos_usados={"hba1c": 7.2},
        regla_id="DX001",
    )
    assert len(ctx.auditoria) == 1
    d = ctx.auditoria[0]
    assert d.accion == "confirmar_dm2"
    assert d.motor == "reglas_duras"
    assert d.regla_id == "DX001"
    assert d.datos_usados == {"hba1c": 7.2}

def test_snapshot_es_json_serializable():
    ctx = _ctx_rural_digital()
    ctx.dm2_confirmado = True
    ctx.riesgo_ml = 0.31
    ctx.via_ruteo = ViaRuteo.AMARILLA
    ctx.log_decision("fase2_ews", "motor_ml", "via_amarilla",
                     "riesgo 0.31", {"riesgo": 0.31})
    snap = ctx.snapshot()
    json_str = json.dumps(snap)
    assert "TEST001" in json_str
    assert "amarilla" in json_str
    assert snap["auditoria"][0]["accion"] == "via_amarilla"
    assert snap["perfil_paciente"]["canal_diagnostico"] == "tele_domicilio"
    assert snap["perfil_paciente"]["canal_monitoreo"] == "whatsapp_autonomo"

def test_snapshot_restauracion_roundtrip():
    ctx = _ctx_rural_digital()
    ctx.dm2_confirmado = True
    ctx.riesgo_ml = 0.42
    ctx.modelo_usado = "xgboost_v1.2.0"
    ctx.via_ruteo = ViaRuteo.AMARILLA
    ctx.barrera_activada = BarreraActivada.MODELO_ML
    ctx.requiere_hitl = True
    ctx.log_decision("fase2_ews", "motor_ml", "via_amarilla",
                     "riesgo 0.42", {"riesgo": 0.42}, regla_id="TX005")

    ctx2 = PacienteContext.restaurar_desde_snapshot(ctx.snapshot())

    assert ctx2.paciente_id == ctx.paciente_id
    assert ctx2.dm2_confirmado is True
    assert ctx2.riesgo_ml == 0.42
    assert ctx2.modelo_usado == "xgboost_v1.2.0"
    assert ctx2.via_ruteo == ViaRuteo.AMARILLA
    assert ctx2.barrera_activada == BarreraActivada.MODELO_ML
    assert ctx2.requiere_hitl is True
    assert ctx2.perfil_paciente.zona == ZonaGeografica.RURAL_AISLADO
    assert ctx2.perfil_paciente.canal_diagnostico == CanalDiagnostico.TELECONSULTA_DOMICILIO
    assert ctx2.perfil_paciente.canal_monitoreo == CanalMonitoreo.WHATSAPP_AUTONOMO
    assert len(ctx2.auditoria) == 1
    assert ctx2.auditoria[0].regla_id == "TX005"

def test_contexto_inicial_no_tiene_barrera_activada():
    ctx = _ctx_urbano_digital()
    assert ctx.barrera_activada == BarreraActivada.NINGUNA
    assert ctx.via_ruteo is None
    assert ctx.riesgo_ml is None
    assert ctx.dm2_confirmado is False
    assert ctx.auditoria == []
    assert ctx.especialista_respaldo_usado is False

def test_log_decision_propaga_paciente_id():
    ctx = PacienteContext(
        paciente_id="P09999",
        perfil_paciente=_ctx_urbano_digital().perfil_paciente,
        fase_actual=_ctx_urbano_digital().fase_actual,
    )
    ctx.log_decision(
        fase="fase_test", motor="motor_test", accion="accion_test",
        justificacion="test de propagacion de paciente_id",
        datos_usados={"clave": "valor"},
    )
    assert ctx.auditoria[-1].paciente_id == "P09999"
