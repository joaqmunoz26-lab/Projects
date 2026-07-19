from dataclasses import dataclass
from enum import Enum


class ZonaGeografica(Enum):
    URBANO        = "urbano"
    RURAL_CERCANO = "rural_cercano"
    RURAL_AISLADO = "rural_aislado"

class CanalDiagnostico(Enum):
    PRESENCIAL_CENTRO      = "presencial_centro"
    TELECONSULTA_DOMICILIO = "tele_domicilio"
    TELECONSULTA_POSTA     = "tele_posta"

class CanalMonitoreo(Enum):
    APP_AUTONOMA      = "app_autonoma"
    WHATSAPP_AUTONOMO = "whatsapp_autonomo"
    TENS_INGRESA      = "tens_ingresa"

@dataclass(frozen=True)
class PerfilPaciente:
    zona:              ZonaGeografica
    canal_diagnostico: CanalDiagnostico
    canal_monitoreo:   CanalMonitoreo
    capacidad_movilizacion_clp_mes: int = 30000
    tramo_fonasa: str = "B"
    acceso_internet_domicilio: bool = True
    presupuesto_plan_datos_clp_mes: int = 8000

    @property
    def tiene_barreras_acceso_fisico(self) -> bool:
        return self.zona in (ZonaGeografica.RURAL_CERCANO, ZonaGeografica.RURAL_AISLADO)

    @property
    def requiere_tens_facilitador(self) -> bool:
        return self.canal_diagnostico == CanalDiagnostico.TELECONSULTA_POSTA

    @property
    def requiere_tens_ingresador(self) -> bool:
        return self.canal_monitoreo == CanalMonitoreo.TENS_INGRESA

    @property
    def es_autonomo_digital(self) -> bool:
        return self.canal_monitoreo in (
            CanalMonitoreo.APP_AUTONOMA,
            CanalMonitoreo.WHATSAPP_AUTONOMO,
        )

    @property
    def descripcion_legible(self) -> str:
        _zona = {
            ZonaGeografica.URBANO:        "Urbano",
            ZonaGeografica.RURAL_CERCANO: "Rural cercano",
            ZonaGeografica.RURAL_AISLADO: "Rural aislado",
        }
        _dx = {
            CanalDiagnostico.PRESENCIAL_CENTRO:      "presencial centro",
            CanalDiagnostico.TELECONSULTA_DOMICILIO: "teleconsulta domicilio",
            CanalDiagnostico.TELECONSULTA_POSTA:     "teleconsulta posta (TENS facilita)",
        }
        _mon = {
            CanalMonitoreo.APP_AUTONOMA:      "app autonoma",
            CanalMonitoreo.WHATSAPP_AUTONOMO: "WhatsApp autonomo",
            CanalMonitoreo.TENS_INGRESA:      "TENS ingresa",
        }
        return (f"{_zona[self.zona]} | "
                f"Dx: {_dx[self.canal_diagnostico]} | "
                f"Monitoreo: {_mon[self.canal_monitoreo]}")

if __name__ == "__main__":
    casos = [
        (ZonaGeografica.URBANO,        CanalDiagnostico.PRESENCIAL_CENTRO,      CanalMonitoreo.APP_AUTONOMA,      "Urbano digital"),
        (ZonaGeografica.RURAL_AISLADO, CanalDiagnostico.TELECONSULTA_DOMICILIO, CanalMonitoreo.WHATSAPP_AUTONOMO, "Rural digital"),
        (ZonaGeografica.RURAL_AISLADO, CanalDiagnostico.TELECONSULTA_POSTA,     CanalMonitoreo.TENS_INGRESA,      "Rural posta"),
        (ZonaGeografica.URBANO,        CanalDiagnostico.PRESENCIAL_CENTRO,      CanalMonitoreo.TENS_INGRESA,      "Urbano mayor"),
    ]
    print("=== PerfilPaciente Demo ===")
    for zona, cd, cm, desc in casos:
        p = PerfilPaciente(zona, cd, cm)
        print(f"\n  {desc}: {p.descripcion_legible}")
        print(f"    barreras={p.tiene_barreras_acceso_fisico} "
              f"tens_facilita={p.requiere_tens_facilitador} "
              f"tens_ingresa={p.requiere_tens_ingresador} "
              f"autonomo={p.es_autonomo_digital}")
