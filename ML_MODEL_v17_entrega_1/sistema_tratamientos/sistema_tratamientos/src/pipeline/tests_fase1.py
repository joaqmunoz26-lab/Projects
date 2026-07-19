import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from core.perfiles_tipicos import (
    perfil_rural_digital,
    perfil_rural_posta,
    perfil_urbano_digital,
)
from pipeline.fase1_ingesta import (
    IngestaPresencialCentro,
    IngestaTeleconsultaDomicilio,
    IngestaTeleconsultaPosta,
    RuteadorIngesta,
    ejecutar_fase1,
)


def _ctx(perfil: PerfilPaciente, datos: dict = None) -> PacienteContext:
    return PacienteContext(
        paciente_id="T",
        perfil_paciente=perfil,
        fase_actual=FaseActual.FASE_1_INGESTA,
        datos_clinicos=datos or {},
    )

def test_urbano_digital_va_a_presencial_centro():
    e = RuteadorIngesta().seleccionar_estrategia(perfil_urbano_digital())
    assert isinstance(e, IngestaPresencialCentro)

def test_rural_digital_cercano_va_a_tele_domicilio():
    e = RuteadorIngesta().seleccionar_estrategia(perfil_rural_digital(aislado=False))
    assert isinstance(e, IngestaTeleconsultaDomicilio)

def test_rural_digital_aislado_va_a_tele_domicilio():
    e = RuteadorIngesta().seleccionar_estrategia(perfil_rural_digital(aislado=True))
    assert isinstance(e, IngestaTeleconsultaDomicilio)

def test_rural_posta_va_a_tele_con_tens_facilitador():
    perfil = perfil_rural_posta()
    e = RuteadorIngesta().seleccionar_estrategia(perfil)
    assert isinstance(e, IngestaTeleconsultaPosta)
    assert perfil.requiere_tens_facilitador is True

def test_urbano_con_tens_ingresador_va_a_presencial_centro():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
    )
    e = RuteadorIngesta().seleccionar_estrategia(perfil)
    assert isinstance(e, IngestaPresencialCentro)

def test_ruteador_mapeo_1a1_por_canal_diagnostico():
    base = ZonaGeografica.URBANO
    mon  = CanalMonitoreo.APP_AUTONOMA
    ruteador = RuteadorIngesta()
    assert isinstance(
        ruteador.seleccionar_estrategia(PerfilPaciente(base, CanalDiagnostico.PRESENCIAL_CENTRO, mon)),
        IngestaPresencialCentro)
    assert isinstance(
        ruteador.seleccionar_estrategia(PerfilPaciente(base, CanalDiagnostico.TELECONSULTA_DOMICILIO, mon)),
        IngestaTeleconsultaDomicilio)
    assert isinstance(
        ruteador.seleccionar_estrategia(PerfilPaciente(base, CanalDiagnostico.TELECONSULTA_POSTA, mon)),
        IngestaTeleconsultaPosta)

def test_rural_posta_requiere_tens_facilitador_y_ingresador():
    p = perfil_rural_posta()
    assert p.requiere_tens_facilitador is True
    assert p.requiere_tens_ingresador is True

def test_rural_digital_no_requiere_tens():
    p = perfil_rural_digital(aislado=True)
    assert p.requiere_tens_facilitador is False
    assert p.requiere_tens_ingresador is False

def test_urbano_digital_es_autonomo_digital():
    p = perfil_urbano_digital()
    assert p.es_autonomo_digital is True


def test_confirmacion_dm2_transiciona_a_fase2():
    ctx = _ctx(perfil_urbano_digital(), {"glucosa_ayunas": 145, "hba1c": 7.2})
    ejecutar_fase1(ctx)
    assert ctx.dm2_confirmado is True
    assert ctx.fecha_confirmacion_dm2 is not None
    assert ctx.fase_actual == FaseActual.FASE_2_EWS

def test_no_confirmacion_dm2_mantiene_fase1():
    ctx = _ctx(perfil_urbano_digital(), {"glucosa_ayunas": 90, "hba1c": 5.5})
    ejecutar_fase1(ctx)
    assert ctx.dm2_confirmado is False
    assert ctx.fase_actual == FaseActual.FASE_1_INGESTA


def test_fase1_registra_precondicion_primera_consulta_presencial():
    ctx = _ctx(perfil_urbano_digital(), {"glucosa_ayunas": 145, "hba1c": 7.2})
    ejecutar_fase1(ctx)
    acciones = [d.accion for d in ctx.auditoria]
    assert "precondicion_primera_consulta_presencial" in acciones
