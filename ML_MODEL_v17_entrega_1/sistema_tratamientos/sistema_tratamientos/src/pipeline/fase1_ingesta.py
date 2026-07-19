import random
import sys
from dataclasses import replace as _dc_replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.examenes_diferidos import CalendarioExamenes
from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)

_GLUCOSA_DM2 = 126.0
_HBA1C_DM2   = 6.5

EXAMENES_DIAGNOSTICOS_GES_DM2 = [
    {"nombre": "hba1c",              "prioridad_clinica": 1,
     "mes_default": 1, "obligatorio_diagnostico": True},
    {"nombre": "glicemia_ayunas",    "prioridad_clinica": 1,
     "mes_default": 1, "obligatorio_diagnostico": True},
    {"nombre": "perfil_lipidico",    "prioridad_clinica": 2,
     "mes_default": 1, "obligatorio_diagnostico": False},
    {"nombre": "microalbuminuria",   "prioridad_clinica": 2,
     "mes_default": 1, "obligatorio_diagnostico": False},
    {"nombre": "funcion_renal_egfr", "prioridad_clinica": 3,
     "mes_default": 1, "obligatorio_diagnostico": False},
    {"nombre": "ecg_basal",          "prioridad_clinica": 3,
     "mes_default": 1, "obligatorio_diagnostico": False},
]

UMBRAL_TRASLADO_PRESENCIAL_CLP = 30_000

UMBRAL_PLAN_DATOS_CLP = 8_000

def asignar_canal_monitoreo(perfil: PerfilPaciente) -> CanalMonitoreo:
    if not perfil.acceso_internet_domicilio:
        return CanalMonitoreo.TENS_INGRESA
    if perfil.zona == ZonaGeografica.RURAL_AISLADO:
        return CanalMonitoreo.TENS_INGRESA
    if perfil.capacidad_movilizacion_clp_mes < UMBRAL_TRASLADO_PRESENCIAL_CLP:
        return CanalMonitoreo.TENS_INGRESA
    if perfil.presupuesto_plan_datos_clp_mes < UMBRAL_PLAN_DATOS_CLP:
        return CanalMonitoreo.TENS_INGRESA
    if perfil.zona == ZonaGeografica.RURAL_CERCANO:
        return CanalMonitoreo.WHATSAPP_AUTONOMO
    return CanalMonitoreo.APP_AUTONOMA

class IngestaStrategy:
    def nombre(self) -> str:
        raise NotImplementedError

    def ingresar(self, ctx: PacienteContext) -> PacienteContext:
        raise NotImplementedError

    def _confirmar_dm2(self, ctx: PacienteContext) -> bool:
        datos = ctx.datos_clinicos
        return ((datos.get("glucosa_ayunas") or 0) >= _GLUCOSA_DM2
                or (datos.get("hba1c") or 0) >= _HBA1C_DM2)

class IngestaPresencialCentro(IngestaStrategy):

    def nombre(self) -> str:
        return "presencial_centro"

    def ingresar(self, ctx: PacienteContext) -> PacienteContext:
        dm2 = self._confirmar_dm2(ctx)
        if dm2:
            ctx.dm2_confirmado        = True
            ctx.fecha_confirmacion_dm2 = datetime.now()

        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ingesta",
            accion="ingesta_presencial_centro",
            justificacion=(
                f"Consulta presencial en centro. "
                f"DM2={'confirmada' if dm2 else 'pendiente'}. "
                f"Zona: {ctx.perfil_paciente.zona.value}"
            ),
            datos_usados={
                "zona":           ctx.perfil_paciente.zona.value,
                "glucosa_ayunas": ctx.datos_clinicos.get("glucosa_ayunas"),
                "hba1c":          ctx.datos_clinicos.get("hba1c"),
            },
        )
        return ctx

class IngestaTeleconsultaDomicilio(IngestaStrategy):

    def nombre(self) -> str:
        return "teleconsulta_domicilio"

    def ingresar(self, ctx: PacienteContext) -> PacienteContext:
        dm2 = self._confirmar_dm2(ctx)
        if dm2:
            ctx.dm2_confirmado        = True
            ctx.fecha_confirmacion_dm2 = datetime.now()

        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ingesta",
            accion="ingesta_teleconsulta_domicilio",
            justificacion=(
                f"Teleconsulta desde domicilio. "
                f"DM2={'confirmada' if dm2 else 'pendiente'}. "
                f"Zona: {ctx.perfil_paciente.zona.value}, "
                f"Monitor: {ctx.perfil_paciente.canal_monitoreo.value}"
            ),
            datos_usados={
                "zona":            ctx.perfil_paciente.zona.value,
                "canal_monitoreo": ctx.perfil_paciente.canal_monitoreo.value,
                "glucosa_ayunas":  ctx.datos_clinicos.get("glucosa_ayunas"),
                "hba1c":           ctx.datos_clinicos.get("hba1c"),
            },
        )
        return ctx

class IngestaTeleconsultaPosta(IngestaStrategy):

    def nombre(self) -> str:
        return "teleconsulta_posta"

    def ingresar(self, ctx: PacienteContext) -> PacienteContext:
        dm2 = self._confirmar_dm2(ctx)
        if dm2:
            ctx.dm2_confirmado        = True
            ctx.fecha_confirmacion_dm2 = datetime.now()

        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="ingesta",
            accion="ingesta_teleconsulta_posta_tens_facilitador",
            justificacion=(
                f"Teleconsulta asistida en posta. TENS facilita equipo tecnologico. "
                f"Consulta es medico-paciente. "
                f"DM2={'confirmada' if dm2 else 'pendiente'}. "
                f"Monitor Fase 2: {ctx.perfil_paciente.canal_monitoreo.value}"
            ),
            datos_usados={
                "zona":              ctx.perfil_paciente.zona.value,
                "tens_facilita_f1":  True,
                "tens_ingresa_f2":   ctx.perfil_paciente.requiere_tens_ingresador,
                "canal_monitoreo":   ctx.perfil_paciente.canal_monitoreo.value,
                "glucosa_ayunas":    ctx.datos_clinicos.get("glucosa_ayunas"),
                "hba1c":             ctx.datos_clinicos.get("hba1c"),
            },
        )
        return ctx


class RuteadorIngesta:
    _MAPEO = {
        CanalDiagnostico.PRESENCIAL_CENTRO:      IngestaPresencialCentro,
        CanalDiagnostico.TELECONSULTA_DOMICILIO: IngestaTeleconsultaDomicilio,
        CanalDiagnostico.TELECONSULTA_POSTA:     IngestaTeleconsultaPosta,
    }

    def seleccionar_estrategia(self, perfil: PerfilPaciente) -> IngestaStrategy:
        clase = self._MAPEO.get(perfil.canal_diagnostico)
        if clase is None:
            raise ValueError(
                f"No hay estrategia para canal_diagnostico={perfil.canal_diagnostico}"
            )
        return clase()

def ejecutar_fase1(ctx: PacienteContext) -> PacienteContext:
    ctx.log_decision(
        fase=ctx.fase_actual.value,
        motor="fase1_ingesta",
        accion="precondicion_primera_consulta_presencial",
        justificacion=(
            "Se asume primera consulta presencial previa donde se levantaron "
            "antecedentes y solicitaron examenes GES"
        ),
        datos_usados={
            "canal_diagnostico_consulta_revision": ctx.perfil_paciente.canal_diagnostico.value
        },
    )

    ruteador   = RuteadorIngesta()
    estrategia = ruteador.seleccionar_estrategia(ctx.perfil_paciente)

    ctx.log_decision(
        fase=ctx.fase_actual.value, motor="ruteador_ingesta",
        accion=f"estrategia_{estrategia.nombre()}",
        justificacion=ctx.perfil_paciente.descripcion_legible,
        datos_usados={
            "zona":              ctx.perfil_paciente.zona.value,
            "canal_diagnostico": ctx.perfil_paciente.canal_diagnostico.value,
            "canal_monitoreo":   ctx.perfil_paciente.canal_monitoreo.value,
        },
    )

    tramo      = ctx.perfil_paciente.tramo_fonasa
    calendario = CalendarioExamenes.crear_para_paciente_ges(
        examenes_recomendados=EXAMENES_DIAGNOSTICOS_GES_DM2,
    )
    ctx.calendario_examenes = calendario

    ctx.log_decision(
        fase=ctx.fase_actual.value, motor="planificacion_examenes_ges",
        accion="calendario_ges_visita_unica",
        justificacion=(
            f"Bateria diagnostica GES DM2 en visita unica de ingreso. Tramo FONASA {tramo}. "
            f"Extraccion de sangre unica cubre glucemia, HbA1c, perfil lipidico y creatinina; "
            f"microalbuminuria en orina simultanea; ECG el mismo dia."
        ),
        datos_usados={
            "tramo_fonasa":                    tramo,
            "examenes_visita_ingreso":         [e.nombre for e in calendario.examenes],
            "meses_hasta_perfil_completo":     calendario.meses_hasta_perfil_completo,
        },
    )

    estrategia.ingresar(ctx)

    if ctx.dm2_confirmado:
        canal_decidido = asignar_canal_monitoreo(ctx.perfil_paciente)
        ctx.perfil_paciente = _dc_replace(ctx.perfil_paciente,
                                          canal_monitoreo=canal_decidido)
        ctx.log_decision(
            fase=ctx.fase_actual.value, motor="asignacion_canal_monitoreo",
            accion=f"canal_asignado_{canal_decidido.value}",
            justificacion=(
                f"Filtro CONEX: internet={ctx.perfil_paciente.acceso_internet_domicilio}. "
                f"Filtro CAP-1 traslado: ${ctx.perfil_paciente.capacidad_movilizacion_clp_mes:,}/mes "
                f"(umbral ${UMBRAL_TRASLADO_PRESENCIAL_CLP:,}). "
                f"Filtro CAP-2 datos: ${ctx.perfil_paciente.presupuesto_plan_datos_clp_mes:,}/mes "
                f"(umbral ${UMBRAL_PLAN_DATOS_CLP:,}). "
                f"Canal asignado: {canal_decidido.value}."
            ),
            datos_usados={
                "tiene_internet":          ctx.perfil_paciente.acceso_internet_domicilio,
                "capacidad_traslado_clp":  ctx.perfil_paciente.capacidad_movilizacion_clp_mes,
                "presupuesto_datos_clp":   ctx.perfil_paciente.presupuesto_plan_datos_clp_mes,
                "canal_resultante":        canal_decidido.value,
            },
        )
        ctx.fase_actual = FaseActual.FASE_2_EWS

    return ctx

if __name__ == "__main__":
    from core.perfiles import CanalMonitoreo, ZonaGeografica
    from core.perfiles_tipicos import (
        perfil_rural_digital,
        perfil_rural_posta,
        perfil_urbano_digital,
    )
    PACIENTES_DEMO = [
        ("P1 Urbano digital",
         perfil_urbano_digital(),
         {"glucosa_ayunas": 145, "hba1c": 7.2}),
        ("P2 Rural digital cercano",
         perfil_rural_digital(aislado=False),
         {"glucosa_ayunas": 155, "hba1c": 7.8}),
        ("P3 Rural digital aislado",
         perfil_rural_digital(aislado=True),
         {"glucosa_ayunas": 90, "hba1c": 5.5}),
        ("P4 Rural posta",
         perfil_rural_posta(),
         {"glucosa_ayunas": 138, "hba1c": 6.9}),
        ("P5 Urbano adulto mayor (TENS ingresa F2)",
         PerfilPaciente(
             zona=ZonaGeografica.URBANO,
             canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
             canal_monitoreo=CanalMonitoreo.TENS_INGRESA,
         ),
         {"glucosa_ayunas": 131, "hba1c": 6.7}),
    ]

    print("=== Demo Fase 1 con modelo bidimensional ===\n")
    for desc, perfil, datos in PACIENTES_DEMO:
        ctx = PacienteContext(
            paciente_id=desc[:2].strip(),
            perfil_paciente=perfil,
            fase_actual=FaseActual.FASE_1_INGESTA,
            datos_clinicos=datos,
        )
        ejecutar_fase1(ctx)
        print(f"  {desc}")
        print(f"    Perfil   : {perfil.descripcion_legible}")
        print(f"    DM2      : {'CONFIRMADA' if ctx.dm2_confirmado else 'pendiente'}")
        print(f"    Fase     : {ctx.fase_actual.value}")
        print("    Auditoria:")
        for d in ctx.auditoria:
            print(f"      [{d.motor}] {d.accion}")
            print(f"        {d.justificacion}")
        print()

    import collections
    DISTRIBUCION = [
        (0.60, lambda: perfil_urbano_digital()),
        (0.25, lambda: perfil_rural_digital(aislado=False)),
        (0.12, lambda: perfil_rural_posta()),
        (0.03, lambda: PerfilPaciente(
            ZonaGeografica.URBANO,
            CanalDiagnostico.PRESENCIAL_CENTRO,
            CanalMonitoreo.TENS_INGRESA,
        )),
    ]
    conteo: dict = collections.Counter()
    N = 1000
    for _ in range(N):
        r = random.random()
        acum = 0.0
        for prob, factory in DISTRIBUCION:
            acum += prob
            if r < acum:
                perfil = factory()
                break
        estrategia = RuteadorIngesta().seleccionar_estrategia(perfil)
        conteo[estrategia.nombre()] += 1

    print("=== Reporte estadistico: 1000 pacientes sinteticos ===")
    print("  Distribucion: 60% urbano digital, 25% rural digital, 12% rural posta, 3% mixtos\n")
    for nombre, cnt in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {nombre:<35} {cnt:>4} ({cnt/N*100:.1f}%)")
