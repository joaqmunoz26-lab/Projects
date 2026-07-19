import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from servicios.hitl_revision import ColaRevisionHITL, EstadoHITL, SolicitudHITL

_RESOLVIBLES_HUMANO = (
    EstadoHITL.PENDIENTE,
    EstadoHITL.EN_REVISION,
)

@dataclass
class EventoAprobacion:
    solicitud_id:       str
    paciente_id:        str
    accion:             str
    medico_id:          str
    fase_protocolo:     str
    horas_transcurridas: float
    dia_simulacion:     int


class SimuladorAprobacionHITL:
    PROB_APROBAR_SLA_NORMAL  = 0.30
    PROB_RECHAZAR_SLA_NORMAL = 0.02

    PROB_APROBAR_FASE_A      = 0.45
    PROB_RECHAZAR_FASE_A     = 0.02

    PROB_APROBAR_FASE_B      = 0.55
    PROB_RECHAZAR_FASE_B     = 0.02

    def __init__(self, seed: int = 42):
        self.rng    = random.Random(seed)
        self.eventos: list[EventoAprobacion] = []
        self.pacientes_reclasificados: set[str] = set()

    def _marcar_reclasificacion_paciente(self, paciente_id: str) -> None:
        self.pacientes_reclasificados.add(paciente_id)

    def es_reclasificado(self, paciente_id: str) -> bool:
        return paciente_id in self.pacientes_reclasificados

    def procesar_dia(
        self,
        cola_hitl:      ColaRevisionHITL,
        dia_simulacion: int,
        ahora:          datetime | None = None,
    ) -> list[EventoAprobacion]:
        if ahora is None:
            ahora = datetime.now() + timedelta(days=dia_simulacion)

        eventos_dia = []

        for solicitud in list(cola_hitl._solicitudes.values()):
            if solicitud.estado not in _RESOLVIBLES_HUMANO:
                continue

            horas = (ahora - solicitud.creada_en).total_seconds() / 3600

            if horas >= 96:
                continue

            evento = self._procesar_solicitud(
                solicitud, cola_hitl,
                horas, dia_simulacion,
            )

            if evento is not None:
                eventos_dia.append(evento)

        self.eventos.extend(eventos_dia)
        return eventos_dia

    def _procesar_solicitud(
        self,
        solicitud:    SolicitudHITL,
        cola_hitl:    ColaRevisionHITL,
        horas:        float,
        dia_simulacion: int,
    ) -> EventoAprobacion | None:
        if horas < 48:
            fase       = "SLA_normal"
            p_aprobar  = self.PROB_APROBAR_SLA_NORMAL
            p_rechazar = self.PROB_RECHAZAR_SLA_NORMAL
        elif horas < 72:
            fase       = "Fase_A_alerta"
            p_aprobar  = self.PROB_APROBAR_FASE_A
            p_rechazar = self.PROB_RECHAZAR_FASE_A
        else:
            fase       = "Fase_B_respaldo"
            p_aprobar  = self.PROB_APROBAR_FASE_B
            p_rechazar = self.PROB_RECHAZAR_FASE_B

        sorteo = self.rng.random()

        if sorteo < p_aprobar:
            medico_id = self._seleccionar_medico(solicitud)
            cola_hitl.aprobar(
                solicitud.id, medico_id=medico_id,
                comentario=f"Aprobado en {fase} (simulacion)",
            )
            return EventoAprobacion(
                solicitud_id=solicitud.id,
                paciente_id=solicitud.paciente_id,
                accion="aprobada",
                medico_id=medico_id,
                fase_protocolo=fase,
                horas_transcurridas=round(horas, 2),
                dia_simulacion=dia_simulacion,
            )

        if sorteo < p_aprobar + p_rechazar:
            medico_id = self._seleccionar_medico(solicitud)
            cola_hitl.rechazar(
                solicitud.id, medico_id=medico_id,
                comentario="Rechazo clinico, escalar a Amarilla (simulacion)",
            )
            self._marcar_reclasificacion_paciente(solicitud.paciente_id)
            return EventoAprobacion(
                solicitud_id=solicitud.id,
                paciente_id=solicitud.paciente_id,
                accion="rechazada",
                medico_id=medico_id,
                fase_protocolo=fase,
                horas_transcurridas=round(horas, 2),
                dia_simulacion=dia_simulacion,
            )

        return None

    def _seleccionar_medico(self, solicitud: SolicitudHITL) -> str:
        return solicitud.especialista_titular or "doc_001"

    def estadisticas(self) -> dict:
        total = len(self.eventos)
        if total == 0:
            return {"total_acciones_humanas": 0}

        aprobadas  = sum(1 for e in self.eventos if e.accion == "aprobada")
        rechazadas = sum(1 for e in self.eventos if e.accion == "rechazada")

        por_fase: dict[str, int] = {}
        for e in self.eventos:
            por_fase[e.fase_protocolo] = por_fase.get(e.fase_protocolo, 0) + 1

        return {
            "total_acciones_humanas": total,
            "aprobadas":              aprobadas,
            "rechazadas":             rechazadas,
            "por_fase":               por_fase,
            "tasa_aprobacion_pct":    round(100 * aprobadas  / total, 2),
            "tasa_rechazo_pct":       round(100 * rechazadas / total, 2),
        }
