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
from motor.reglas_duras import MotorReglasDuras


def _ctx(datos: dict) -> PacienteContext:
    return PacienteContext(
        paciente_id="T001",
        perfil_paciente=PerfilPaciente(
            zona=ZonaGeografica.URBANO,
            canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        datos_clinicos=datos,
    )

def test_glucosa_350_activa_rd001():
    ctx = _ctx({"glucosa_ayunas": 350})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD001"
    assert r.severidad == "CRITICA"
    assert ctx.barrera_activada == BarreraActivada.REGLAS_DURAS
    assert ctx.via_ruteo == ViaRuteo.ROJA
    assert len(ctx.auditoria) == 1
    assert ctx.auditoria[0].motor == "reglas_duras"

def test_glucosa_50_activa_rd002():
    ctx = _ctx({"glucosa_ayunas": 50})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD002"
    assert r.severidad == "CRITICA"
    assert ctx.barrera_activada == BarreraActivada.REGLAS_DURAS

def test_hba1c_12_activa_rd003():
    ctx = _ctx({"hba1c": 12.0})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD003"
    assert r.severidad == "ALTA"
    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_pas_190_activa_rd004():
    ctx = _ctx({"presion_sistolica": 190, "presion_diastolica": 85})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD004"
    assert r.severidad == "CRITICA"

def test_egfr_25_activa_rd005():
    ctx = _ctx({"funcion_renal_egfr": 25})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD005"
    assert r.severidad == "ALTA"

def test_paciente_normal_no_activa_ninguna():
    ctx = _ctx({"glucosa_ayunas": 115, "hba1c": 7.2,
                "presion_sistolica": 125, "presion_diastolica": 80,
                "funcion_renal_egfr": 68})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is False
    assert r.regla_activada is None
    assert ctx.barrera_activada == BarreraActivada.NINGUNA
    assert ctx.via_ruteo is None
    assert len(ctx.auditoria) == 1
    assert ctx.auditoria[0].accion == "evaluacion_sin_matcheo"

def test_multiples_reglas_activas_gana_criticidad_mayor():
    ctx = _ctx({"glucosa_ayunas": 350, "hba1c": 12.0})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD001"
    assert r.severidad == "CRITICA"
    assert len(ctx.auditoria) == 1

def test_variables_faltantes_no_crashea():
    ctx = _ctx({})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is False
    assert ctx.barrera_activada == BarreraActivada.NINGUNA

    ctx2 = _ctx({"hba1c": 12.0})
    r2 = MotorReglasDuras().evaluar(ctx2)
    assert r2.matcheo is True
    assert r2.regla_activada == "RD003"

def test_rd004_umbral_exacto_pas_180_activa():
    ctx = _ctx({"presion_sistolica": 180, "presion_diastolica": 80})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD004"
    assert r.severidad == "CRITICA"
    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_rd004_umbral_exacto_pad_110_activa():
    ctx = _ctx({"presion_sistolica": 140, "presion_diastolica": 110})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD004"
    assert r.severidad == "CRITICA"
    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_rd004_un_punto_bajo_umbral_no_activa():
    ctx = _ctx({"presion_sistolica": 179, "presion_diastolica": 109})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is False
    assert r.regla_activada is None


def test_rd004_solo_pas_190_sin_pad_activa():
    ctx = _ctx({"presion_sistolica": 190})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD004"
    assert r.severidad == "CRITICA"
    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_rd004_solo_pad_115_sin_pas_activa():
    ctx = _ctx({"presion_diastolica": 115})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is True
    assert r.regla_activada == "RD004"
    assert r.severidad == "CRITICA"
    assert ctx.via_ruteo == ViaRuteo.ROJA

def test_rd004_solo_pas_bajo_umbral_sin_pad_no_activa():
    ctx = _ctx({"presion_sistolica": 150})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is False
    assert r.regla_activada is None

def test_rd004_ambas_ausentes_no_activa():
    ctx = _ctx({"glucosa_ayunas": 115, "hba1c": 7.2})
    r = MotorReglasDuras().evaluar(ctx)
    assert r.matcheo is False
    assert r.regla_activada is None

def test_main_reglas_duras_corre():
    import subprocess
    script = Path(__file__).resolve().parent / "reglas_duras.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
