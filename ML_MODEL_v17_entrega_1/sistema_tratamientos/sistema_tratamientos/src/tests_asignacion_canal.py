import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from pipeline.fase1_ingesta import asignar_canal_monitoreo


def test_sin_internet_asigna_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=False,
        capacidad_movilizacion_clp_mes=50000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_con_internet_baja_capacidad_asigna_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.RURAL_CERCANO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_DOMICILIO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=8000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_con_internet_alta_capacidad_urbano_asigna_app():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=40000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.APP_AUTONOMA

def test_con_internet_alta_capacidad_rural_cercano_asigna_whatsapp():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.RURAL_CERCANO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_DOMICILIO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=35000,
        presupuesto_plan_datos_clp_mes=10000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.WHATSAPP_AUTONOMO

def test_con_internet_alta_capacidad_rural_aislado_asigna_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=15000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_umbral_exacto_traslado():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=30000,
        presupuesto_plan_datos_clp_mes=10000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.APP_AUTONOMA

def test_umbral_justo_bajo_traslado():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=29999,
        presupuesto_plan_datos_clp_mes=10000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_rural_aislado_con_filtros_ok_recibe_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=50_000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_urbano_con_filtros_ok_recibe_app():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=50_000,
        presupuesto_plan_datos_clp_mes=10_000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.APP_AUTONOMA

def test_cap_traslado_insuficiente_va_a_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=20_000,
        presupuesto_plan_datos_clp_mes=10_000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_cap_plan_datos_insuficiente_va_a_tens():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=40_000,
        presupuesto_plan_datos_clp_mes=5_000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.TENS_INGRESA

def test_cap_ambos_suficientes_va_a_canal_digital():
    perfil = PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        acceso_internet_domicilio=True,
        capacidad_movilizacion_clp_mes=40_000,
        presupuesto_plan_datos_clp_mes=10_000,
    )
    assert asignar_canal_monitoreo(perfil) == CanalMonitoreo.APP_AUTONOMA
