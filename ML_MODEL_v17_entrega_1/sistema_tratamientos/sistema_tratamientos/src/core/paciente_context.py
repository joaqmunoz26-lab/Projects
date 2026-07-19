from dataclasses import dataclass, field
from datetime import datetime

from core.decisiones import BarreraActivada, ViaRuteo
from core.eventos import DecisionAuditada
from core.examenes_diferidos import CalendarioExamenes
from core.fases import FaseActual
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)


@dataclass
class PacienteContext:

    paciente_id:      str
    perfil_paciente:  PerfilPaciente
    fase_actual:      FaseActual

    datos_clinicos:        dict  = field(default_factory=dict)
    historial_controles:   list  = field(default_factory=list)

    fecha_proximo_control_rutina: datetime | None = None
    control_rutina_adelantado:    bool = False

    dm2_confirmado:         bool               = False
    fecha_confirmacion_dm2: datetime | None = None

    riesgo_ml:        float | None = None
    modelo_usado:     str | None   = None
    barrera_activada: BarreraActivada = BarreraActivada.NINGUNA
    via_ruteo:        ViaRuteo | None = None

    requiere_hitl:            bool          = False
    hitl_solicitud_id:        str | None = None
    aprobacion_hitl_recibida: bool          = False

    calendario_examenes:      CalendarioExamenes | None = None

    especialista_titular:        str | None = None
    especialista_asignado:       str | None = None
    especialista_respaldo_usado: bool          = False

    auditoria: list = field(default_factory=list)

    def log_decision(self, fase: str, motor: str, accion: str,
                     justificacion: str, datos_usados: dict,
                     regla_id: str | None = None,
                     usuario_id: str | None = None) -> None:
        self.auditoria.append(DecisionAuditada(
            timestamp=datetime.now(),
            fase=fase, motor=motor, accion=accion,
            justificacion=justificacion, datos_usados=datos_usados,
            regla_id=regla_id, usuario_id=usuario_id,
            paciente_id=self.paciente_id,
        ))

    def reclasificar_a_via_amarilla(self, motivo: str) -> None:

        via_previa = self.via_ruteo
        self.via_ruteo = ViaRuteo.AMARILLA
        self.control_rutina_adelantado = True
        self.log_decision(
            fase=self.fase_actual.value,
            motor="hitl_revision",
            accion="reclasificacion_via_amarilla_hitl_c",
            justificacion=motivo,
            datos_usados={
                "via_previa": via_previa.value if via_previa else None,
                "via_nueva":  ViaRuteo.AMARILLA.value,
            },
        )

    def snapshot(self) -> dict:
        p = self.perfil_paciente
        return {
            "paciente_id": self.paciente_id,
            "perfil_paciente": {
                "zona":                           p.zona.value,
                "canal_diagnostico":              p.canal_diagnostico.value,
                "canal_monitoreo":                p.canal_monitoreo.value,
                "capacidad_movilizacion_clp_mes": p.capacidad_movilizacion_clp_mes,
                "presupuesto_plan_datos_clp_mes": p.presupuesto_plan_datos_clp_mes,
                "tramo_fonasa":                   p.tramo_fonasa,
                "acceso_internet_domicilio":      p.acceso_internet_domicilio,
            },
            "fase_actual":     self.fase_actual.value,
            "datos_clinicos":       self.datos_clinicos,
            "historial_controles":  self.historial_controles,
            "fecha_proximo_control_rutina": (
                self.fecha_proximo_control_rutina.isoformat()
                if self.fecha_proximo_control_rutina else None),
            "control_rutina_adelantado": self.control_rutina_adelantado,
            "dm2_confirmado":        self.dm2_confirmado,
            "fecha_confirmacion_dm2": (
                self.fecha_confirmacion_dm2.isoformat()
                if self.fecha_confirmacion_dm2 else None),
            "riesgo_ml":        self.riesgo_ml,
            "modelo_usado":     self.modelo_usado,
            "barrera_activada": self.barrera_activada.value,
            "via_ruteo":        self.via_ruteo.value if self.via_ruteo else None,
            "requiere_hitl":             self.requiere_hitl,
            "hitl_solicitud_id":         self.hitl_solicitud_id,
            "aprobacion_hitl_recibida":  self.aprobacion_hitl_recibida,
            "especialista_titular":      self.especialista_titular,
            "especialista_asignado":     self.especialista_asignado,
            "especialista_respaldo_usado": self.especialista_respaldo_usado,
            "auditoria": [d.a_dict() for d in self.auditoria],
        }

    @classmethod
    def restaurar_desde_snapshot(cls, data: dict) -> "PacienteContext":
        def _dt(v): return datetime.fromisoformat(v) if v else None

        pd = data["perfil_paciente"]
        perfil = PerfilPaciente(
            zona=ZonaGeografica(pd["zona"]),
            canal_diagnostico=CanalDiagnostico(pd["canal_diagnostico"]),
            canal_monitoreo=CanalMonitoreo(pd["canal_monitoreo"]),
            capacidad_movilizacion_clp_mes=pd.get("capacidad_movilizacion_clp_mes", 30000),
            presupuesto_plan_datos_clp_mes=pd.get("presupuesto_plan_datos_clp_mes", 8000),
            tramo_fonasa=pd.get("tramo_fonasa", "B"),
            acceso_internet_domicilio=pd.get("acceso_internet_domicilio", True),
        )
        auditoria = [
            DecisionAuditada(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                fase=d["fase"], motor=d["motor"], accion=d["accion"],
                justificacion=d["justificacion"], datos_usados=d["datos_usados"],
                regla_id=d.get("regla_id"), usuario_id=d.get("usuario_id"),
                paciente_id=d.get("paciente_id", ""),
            )
            for d in data.get("auditoria", [])
        ]
        return cls(
            paciente_id=data["paciente_id"],
            perfil_paciente=perfil,
            fase_actual=FaseActual(data["fase_actual"]),
            datos_clinicos=data.get("datos_clinicos", {}),
            historial_controles=data.get("historial_controles", []),
            fecha_proximo_control_rutina=_dt(data.get("fecha_proximo_control_rutina")),
            control_rutina_adelantado=data.get("control_rutina_adelantado", False),
            dm2_confirmado=data.get("dm2_confirmado", False),
            fecha_confirmacion_dm2=_dt(data.get("fecha_confirmacion_dm2")),
            riesgo_ml=data.get("riesgo_ml"),
            modelo_usado=data.get("modelo_usado"),
            barrera_activada=BarreraActivada(data.get("barrera_activada", "ninguna")),
            via_ruteo=ViaRuteo(data["via_ruteo"]) if data.get("via_ruteo") else None,
            requiere_hitl=data.get("requiere_hitl", False),
            hitl_solicitud_id=data.get("hitl_solicitud_id"),
            aprobacion_hitl_recibida=data.get("aprobacion_hitl_recibida", False),
            especialista_titular=data.get("especialista_titular"),
            especialista_asignado=data.get("especialista_asignado"),
            especialista_respaldo_usado=data.get("especialista_respaldo_usado", False),
            auditoria=auditoria,
        )
