import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paciente_context import PacienteContext
from core.perfiles import ZonaGeografica


class EstadoHITL(Enum):
    PENDIENTE           = "pendiente"
    EN_REVISION         = "en_revision"
    APROBADA            = "aprobada"
    RECHAZADA           = "rechazada"
    EXPIRADA_ESCALADA   = "expirada_escalada"
    EXPIRADA_EMERGENCIA = "expirada_emergencia"

class TipoSolicitudHITL(Enum):
    ORDEN_VERDE            = "orden_verde"
    REVISION_EXAMENES      = "revision_examenes"
    VALIDACION_DIAGNOSTICO = "validacion_diagnostico"

class PrioridadHITL(Enum):
    NORMAL  = "normal"
    ALTA    = "alta"
    URGENTE = "urgente"

_ORDEN_PRIORIDAD = {
    PrioridadHITL.NORMAL:  1,
    PrioridadHITL.ALTA:    2,
    PrioridadHITL.URGENTE: 3,
}

@dataclass
class SolicitudHITL:
    id:                     str
    paciente_id:            str
    tipo:                   TipoSolicitudHITL
    prioridad:              PrioridadHITL
    datos_contexto:         dict
    riesgo_ml:              float | None
    justificacion_ml:       str
    creada_en:              datetime
    sla_horas:              int = 72
    estado:                 EstadoHITL = EstadoHITL.PENDIENTE
    especialista_titular:   str | None = None
    especialista_aprobador: str | None = None
    fecha_aprobacion:       datetime | None = None
    fecha_rechazo:          datetime | None = None
    comentario_medico:      str | None = None
    escalamientos:          list = field(default_factory=list)
    audit_callback:         Callable | None = field(default=None, compare=False, repr=False)
    callback_reclasificar:  Callable | None = field(default=None, compare=False, repr=False)

    @property
    def horas_transcurridas(self) -> float:
        fecha_final = (self.fecha_aprobacion or self.fecha_rechazo
                       or datetime.now())
        return (fecha_final - self.creada_en).total_seconds() / 3600

    @property
    def esta_activa(self) -> bool:
        return self.estado in (EstadoHITL.PENDIENTE, EstadoHITL.EN_REVISION)


@dataclass
class EventoEscalamiento:
    solicitud_id:            str
    fase_protocolo:          str
    ejecutada_en:            datetime
    resultado:               str
    especialista_notificado: str | None = None
    especialista_reasignado: str | None = None
    datos_extra:             dict = field(default_factory=dict)

class ErrorHITL(Exception):
    pass

class ColaRevisionHITL:
    DIAS_MINIMOS_ANTICIPACION = 5
    def __init__(self):
        self._solicitudes: dict[str, SolicitudHITL] = {}

    def encolar(
        self,
        ctx:                        PacienteContext,
        tipo:                       TipoSolicitudHITL,
        dias_hasta_fin_medicamento: int | None = None,
        especialista_titular:       str | None = None,
    ) -> SolicitudHITL:
        solicitud_id = f"hitl_{uuid.uuid4().hex[:12]}"

        prioridad = PrioridadHITL.NORMAL
        sla_horas = 72

        if (dias_hasta_fin_medicamento is not None
                and dias_hasta_fin_medicamento < self.DIAS_MINIMOS_ANTICIPACION):
            prioridad = PrioridadHITL.ALTA
            sla_horas = 24

        riesgo = ctx.riesgo_ml
        via    = ctx.via_ruteo.value if ctx.via_ruteo else "N/A"

        solicitud = SolicitudHITL(
            id=solicitud_id,
            paciente_id=ctx.paciente_id,
            tipo=tipo,
            prioridad=prioridad,
            datos_contexto={
                "perfil": (ctx.perfil_paciente.descripcion_legible
                           if hasattr(ctx.perfil_paciente, "descripcion_legible")
                           else str(ctx.perfil_paciente)),
                "zona_paciente":    ctx.perfil_paciente.zona.value,
                "via_ruteo":        via,
                "barrera_activada": ctx.barrera_activada.value if ctx.barrera_activada else None,
                "dias_hasta_fin_medicamento": dias_hasta_fin_medicamento,
            },
            riesgo_ml=riesgo,
            justificacion_ml=(
                f"Riesgo ML: {riesgo:.3f} - Via: {via}"
                if riesgo is not None else f"Sin riesgo ML - Via: {via}"
            ),
            creada_en=datetime.now(),
            sla_horas=sla_horas,
            especialista_titular=especialista_titular,
        )

        solicitud.audit_callback        = ctx.log_decision
        solicitud.callback_reclasificar = ctx.reclasificar_a_via_amarilla
        self._solicitudes[solicitud_id] = solicitud

        ctx.requiere_hitl     = True
        ctx.hitl_solicitud_id = solicitud_id
        ctx.log_decision(
            fase="fase2",
            motor="hitl_encolar",
            accion=f"solicitud_encolada_{tipo.value}",
            justificacion=f"Prioridad {prioridad.value}, SLA {sla_horas}h",
            datos_usados={
                "solicitud_id": solicitud_id,
                "tipo":         tipo.value,
                "dias_hasta_fin_medicamento": dias_hasta_fin_medicamento,
            },
        )
        return solicitud

    def obtener(self, solicitud_id: str) -> SolicitudHITL | None:
        return self._solicitudes.get(solicitud_id)

    def obtener_pendientes(
        self,
        especialista_id:  str | None           = None,
        prioridad_minima: PrioridadHITL | None = None,
    ) -> list[SolicitudHITL]:
        nivel_minimo = _ORDEN_PRIORIDAD[prioridad_minima] if prioridad_minima else 0
        resultado = [
            s for s in self._solicitudes.values()
            if s.esta_activa
            and _ORDEN_PRIORIDAD[s.prioridad] >= nivel_minimo
            and (especialista_id is None
                 or s.especialista_titular == especialista_id)
        ]
        resultado.sort(key=lambda s: (-_ORDEN_PRIORIDAD[s.prioridad], s.creada_en))
        return resultado

    def tomar_en_revision(self, solicitud_id: str, medico_id: str) -> SolicitudHITL:
        s = self._get_o_error(solicitud_id)
        if s.estado == EstadoHITL.EN_REVISION:
            raise ErrorHITL(f"Solicitud {solicitud_id} ya esta EN_REVISION")
        if not s.esta_activa:
            raise ErrorHITL(
                f"Solicitud {solicitud_id} en estado {s.estado.value}, "
                "no se puede tomar para revision"
            )
        s.estado = EstadoHITL.EN_REVISION
        s.especialista_aprobador = medico_id
        return s

    _ESTADOS_RESOLVIBLES = (
        EstadoHITL.PENDIENTE,
        EstadoHITL.EN_REVISION,
    )

    def aprobar(self, solicitud_id: str, medico_id: str, comentario: str) -> SolicitudHITL:
        s = self._get_o_error(solicitud_id)
        if s.estado not in self._ESTADOS_RESOLVIBLES:
            raise ErrorHITL(
                f"Solicitud {solicitud_id} esta en estado "
                f"{s.estado.value}, no puede aprobarse"
            )
        s.estado                 = EstadoHITL.APROBADA
        s.especialista_aprobador = medico_id
        s.fecha_aprobacion       = datetime.now()
        s.comentario_medico      = comentario
        if s.audit_callback is not None:
            s.audit_callback(
                fase="fase2", motor="hitl_revision",
                accion="hitl_aprobada",
                justificacion=f"Solicitud {s.id} aprobada por {medico_id}: {comentario}",
                datos_usados={
                    "solicitud_id":  s.id,
                    "tipo_solicitud": s.tipo.value,
                    "estado_final":   s.estado.value,
                    "usuario_id":    medico_id,
                },
                usuario_id=medico_id,
            )
        return s

    def rechazar(self, solicitud_id: str, medico_id: str, comentario: str) -> SolicitudHITL:
        s = self._get_o_error(solicitud_id)
        if s.estado not in self._ESTADOS_RESOLVIBLES:
            raise ErrorHITL(
                f"Solicitud {solicitud_id} esta en estado "
                f"{s.estado.value}, no puede rechazarse"
            )
        s.estado                 = EstadoHITL.RECHAZADA
        s.especialista_aprobador = medico_id
        s.fecha_rechazo          = datetime.now()
        s.comentario_medico      = comentario
        s.datos_contexto["requiere_reclasificacion_via_amarilla"] = True
        s.datos_contexto["motivo_rechazo"]                        = comentario
        if s.audit_callback is not None:
            s.audit_callback(
                fase="fase2", motor="hitl_revision",
                accion="hitl_rechazada",
                justificacion=f"Solicitud {s.id} rechazada por {medico_id}: {comentario}",
                datos_usados={
                    "solicitud_id":  s.id,
                    "tipo_solicitud": s.tipo.value,
                    "estado_final":   s.estado.value,
                    "usuario_id":    medico_id,
                },
                usuario_id=medico_id,
            )
        return s

    def estadisticas(self) -> dict:
        todas = list(self._solicitudes.values())

        por_estado    = {e.value: 0 for e in EstadoHITL}
        por_tipo      = {t.value: 0 for t in TipoSolicitudHITL}
        por_prioridad = {p.value: 0 for p in PrioridadHITL}
        for s in todas:
            por_estado[s.estado.value]       += 1
            por_tipo[s.tipo.value]           += 1
            por_prioridad[s.prioridad.value] += 1

        resueltas = [s for s in todas
                     if s.estado in (EstadoHITL.APROBADA, EstadoHITL.RECHAZADA)]
        tiempo_prom = (
            round(sum(s.horas_transcurridas for s in resueltas) / len(resueltas), 2)
            if resueltas else 0.0
        )

        aprobadas      = por_estado[EstadoHITL.APROBADA.value]
        rechazadas     = por_estado[EstadoHITL.RECHAZADA.value]
        total_resueltas = aprobadas + rechazadas
        tasa_aprobacion = (
            round(aprobadas / total_resueltas * 100, 1) if total_resueltas else 0.0
        )

        return {
            "total_solicitudes":                len(todas),
            "por_estado":                       por_estado,
            "por_tipo":                         por_tipo,
            "por_prioridad":                    por_prioridad,
            "tiempo_promedio_resolucion_horas": tiempo_prom,
            "tasa_aprobacion_pct":              tasa_aprobacion,
        }

    def _get_o_error(self, solicitud_id: str) -> SolicitudHITL:
        s = self._solicitudes.get(solicitud_id)
        if s is None:
            raise ErrorHITL(f"Solicitud {solicitud_id} no existe")
        return s

class MotorEscalamientoHITL:
    def __init__(self, cola_hitl: ColaRevisionHITL):
        self.cola   = cola_hitl
        self.eventos: list[EventoEscalamiento] = []

    def procesar_escalamientos(
        self, ahora: datetime | None = None
    ) -> list[EventoEscalamiento]:
        ahora = ahora or datetime.now()
        eventos_corrida: list[EventoEscalamiento] = []

        _terminales = (EstadoHITL.APROBADA,
                       EstadoHITL.RECHAZADA,
                       EstadoHITL.EXPIRADA_EMERGENCIA)

        for solicitud in list(self.cola._solicitudes.values()):
            if solicitud.estado in _terminales:
                continue

            horas = (ahora - solicitud.creada_en).total_seconds() / 3600

            if (horas >= 48 and horas < 72
                    and "A_preventiva" not in solicitud.escalamientos):
                if solicitud.esta_activa:
                    eventos_corrida.append(self._ejecutar_fase_a(solicitud, ahora))

            elif (horas >= 72 and horas < 96
                    and "B_pool" not in solicitud.escalamientos):
                if solicitud.esta_activa:
                    eventos_corrida.append(self._ejecutar_fase_b(solicitud, ahora))

            elif (horas >= 96
                    and "C_emergencia" not in solicitud.escalamientos):
                if solicitud.esta_activa:
                    eventos_corrida.append(self._ejecutar_fase_c(solicitud, ahora))

        self.eventos.extend(eventos_corrida)
        return eventos_corrida

    def _ejecutar_fase_a(
        self, solicitud: SolicitudHITL, ahora: datetime
    ) -> EventoEscalamiento:
        solicitud.escalamientos.append("A_preventiva")
        horas_restantes = solicitud.sla_horas - (ahora - solicitud.creada_en).total_seconds() / 3600
        return EventoEscalamiento(
            solicitud_id=solicitud.id,
            fase_protocolo="A_preventiva",
            ejecutada_en=ahora,
            resultado="alerta_enviada",
            especialista_notificado=solicitud.especialista_titular,
            datos_extra={
                "mensaje": f"SLA vence en {horas_restantes:.1f}h",
                "prioridad_original": solicitud.prioridad.value,
            },
        )

    def _ejecutar_fase_b(
        self, solicitud: SolicitudHITL, ahora: datetime
    ) -> EventoEscalamiento:
        solicitud.escalamientos.append("B_pool")
        solicitud.prioridad = PrioridadHITL.ALTA
        horas_transcurridas = (ahora - solicitud.creada_en).total_seconds() / 3600
        solicitud.sla_horas = round(horas_transcurridas) + 24
        return EventoEscalamiento(
            solicitud_id=solicitud.id,
            fase_protocolo="B_pool",
            ejecutada_en=ahora,
            resultado="sla_extendido_24h",
            datos_extra={
                "nuevo_sla_horas": solicitud.sla_horas,
                "nueva_prioridad": "alta",
            },
        )

    def _ejecutar_fase_c(
        self, solicitud: SolicitudHITL, ahora: datetime
    ) -> EventoEscalamiento:
        solicitud.escalamientos.append("C_emergencia")
        solicitud.estado = EstadoHITL.EXPIRADA_EMERGENCIA
        solicitud.datos_contexto.update({
            "receta_emergencia_emitida":          True,
            "dias_receta_emergencia":             14,
            "requiere_teleconsulta_regularizacion": True,
            "reclasificado_via_amarilla":         True,
        })

        if solicitud.callback_reclasificar is not None:
            solicitud.callback_reclasificar(
                motivo=f"T+96h sin resolucion HITL, solicitud {solicitud.id}"
            )

        horas_tot = (ahora - solicitud.creada_en).total_seconds() / 3600
        return EventoEscalamiento(
            solicitud_id=solicitud.id,
            fase_protocolo="C_emergencia",
            ejecutada_en=ahora,
            resultado="fallback_emergencia_activado",
            datos_extra={
                "receta_dias":                        14,
                "accion_siguiente":                   "teleconsulta_regularizacion",
                "requiere_intervencion_administracion": True,
                "horas_transcurridas":                round(horas_tot, 1),
            },
        )

    def estadisticas_escalamientos(self) -> dict:
        por_fase   = {"A_preventiva": 0, "B_pool": 0, "C_emergencia": 0}
        b_exitosas = 0
        b_fallidas = 0
        fase_c     = 0

        for e in self.eventos:
            if e.fase_protocolo in por_fase:
                por_fase[e.fase_protocolo] += 1
            if e.fase_protocolo == "B_pool":
                if e.resultado == "sla_extendido_24h":
                    b_exitosas += 1
                else:
                    b_fallidas += 1
            if e.fase_protocolo == "C_emergencia":
                fase_c += 1

        b_total = b_exitosas + b_fallidas
        tasa_b  = round(b_exitosas / b_total * 100, 1) if b_total else 0.0

        return {
            "total_escalamientos":              len(self.eventos),
            "por_fase":                         por_fase,
            "tasa_escalamiento_b_exitosa":      tasa_b,
            "solicitudes_que_llegaron_a_fase_c": fase_c,
        }

if __name__ == "__main__":
    from core.decisiones import BarreraActivada, ViaRuteo
    from core.fases import FaseActual
    from core.perfiles import CanalDiagnostico, CanalMonitoreo, PerfilPaciente

    def _ctx(pid, riesgo, zona=ZonaGeografica.URBANO):
        return PacienteContext(
            paciente_id=pid,
            perfil_paciente=PerfilPaciente(
                zona=zona,
                canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
                canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
            ),
            fase_actual=FaseActual.FASE_2_EWS,
            riesgo_ml=riesgo,
            via_ruteo=ViaRuteo.VERDE,
            barrera_activada=BarreraActivada.NINGUNA,
        )

    cola  = ColaRevisionHITL()
    motor = MotorEscalamientoHITL(cola)

    print("=== Demo protocolo escalamiento HITL T+48/T+72/T+96 ===\n")

    s1 = cola.encolar(_ctx("P_001", 0.15), TipoSolicitudHITL.ORDEN_VERDE,
                      dias_hasta_fin_medicamento=10, especialista_titular="doc_001")
    t0 = s1.creada_en
    print(f"Solicitud encolada: {s1.id[:20]} titular=doc_001")

    print("\nT+10h - sin escalamiento:")
    ev = motor.procesar_escalamientos(ahora=t0 + timedelta(hours=10))
    print(f"  eventos={len(ev)}  estado={s1.estado.value}")

    print("\nT+49h - Fase A (alerta preventiva):")
    ev = motor.procesar_escalamientos(ahora=t0 + timedelta(hours=49))
    print(f"  eventos={len(ev)}  fase={ev[0].fase_protocolo}  "
          f"resultado={ev[0].resultado}  estado={s1.estado.value}")

    print("\nT+73h - Fase B (concesion 24h adicionales):")
    ev = motor.procesar_escalamientos(ahora=t0 + timedelta(hours=73))
    print(f"  eventos={len(ev)}  fase={ev[0].fase_protocolo}  "
          f"resultado={ev[0].resultado}  nuevo_sla_horas={ev[0].datos_extra['nuevo_sla_horas']}")
    print(f"  prioridad={s1.prioridad.value}  estado={s1.estado.value}")

    print("\nT+97h - Fase C (receta emergencia 14 dias + Via Amarilla):")
    ev = motor.procesar_escalamientos(ahora=t0 + timedelta(hours=97))
    print(f"  eventos={len(ev)}  fase={ev[0].fase_protocolo}  resultado={ev[0].resultado}")
    print(f"  estado={s1.estado.value}")
    print(f"  receta_emergencia={s1.datos_contexto['receta_emergencia_emitida']}  "
          f"dias={s1.datos_contexto['dias_receta_emergencia']}")
    print(f"  reclasificado_via_amarilla={s1.datos_contexto['reclasificado_via_amarilla']}")

    print("\n=== Estadisticas escalamientos ===")
    st = motor.estadisticas_escalamientos()
    print(f"  total_escalamientos:      {st['total_escalamientos']}")
    print(f"  por_fase:                 {st['por_fase']}")
    print(f"  tasa_B_exitosa:           {st['tasa_escalamiento_b_exitosa']}%")
    print(f"  llegaron_a_fase_C:        {st['solicitudes_que_llegaron_a_fase_c']}")
