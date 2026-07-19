import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)


def perfil_urbano_digital() -> PerfilPaciente:
    return PerfilPaciente(
        zona=ZonaGeografica.URBANO,
        canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
        canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
    )

def perfil_rural_digital(aislado: bool = False) -> PerfilPaciente:
    zona      = ZonaGeografica.RURAL_AISLADO if aislado else ZonaGeografica.RURAL_CERCANO
    monitoreo = CanalMonitoreo.WHATSAPP_AUTONOMO if aislado else CanalMonitoreo.APP_AUTONOMA
    return PerfilPaciente(
        zona=zona,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_DOMICILIO,
        canal_monitoreo=monitoreo,
    )


def perfil_rural_posta() -> PerfilPaciente:
    return PerfilPaciente(
        zona=ZonaGeografica.RURAL_AISLADO,
        canal_diagnostico=CanalDiagnostico.TELECONSULTA_POSTA,
        canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
    )


if __name__ == "__main__":
    perfiles = [
        ("Urbano digital",        perfil_urbano_digital()),
        ("Rural digital cercano", perfil_rural_digital(aislado=False)),
        ("Rural digital aislado", perfil_rural_digital(aislado=True)),
        ("Rural posta",           perfil_rural_posta()),
    ]
    print("=== Perfiles tipicos del EWS ===")
    for nombre, p in perfiles:
        print(f"\n  {nombre}")
        print(f"    {p.descripcion_legible}")
        print(f"    barreras={p.tiene_barreras_acceso_fisico} | "
              f"tens_facilita={p.requiere_tens_facilitador} | "
              f"tens_ingresa={p.requiere_tens_ingresador} | "
              f"autonomo={p.es_autonomo_digital}")
