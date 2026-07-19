import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.decisiones import BarreraActivada, ViaRuteo
from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from servicios.hitl_revision import (
    ColaRevisionHITL,
    ErrorHITL,
    EstadoHITL,
    MotorEscalamientoHITL,
    PrioridadHITL,
    TipoSolicitudHITL,
)


def _avanzar_horas(dt: datetime, horas: float) -> datetime:
    return dt + timedelta(hours=horas)


def _crear_ctx_mock(
    paciente_id: str = "P_TEST",
    riesgo: float = 0.20,
    via: ViaRuteo = ViaRuteo.VERDE,
    zona: ZonaGeografica = ZonaGeografica.URBANO,
) -> PacienteContext:
    return PacienteContext(
        paciente_id=paciente_id,
        perfil_paciente=PerfilPaciente(
            zona=zona,
            canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        riesgo_ml=riesgo,
        via_ruteo=via,
        barrera_activada=BarreraActivada.NINGUNA,
    )

@pytest.fixture
def cola():
    return ColaRevisionHITL()

def test_encolar_crea_solicitud_con_id_unico(cola):
    ctx = _crear_ctx_mock()
    s1 = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    ctx2 = _crear_ctx_mock("P002")
    s2 = cola.encolar(ctx2, TipoSolicitudHITL.ORDEN_VERDE)
    assert s1.id != s2.id
    assert s1.id.startswith("hitl_")

def test_encolar_orden_verde_con_anticipacion_suficiente_sla_72h(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     dias_hasta_fin_medicamento=10)
    assert s.prioridad == PrioridadHITL.NORMAL
    assert s.sla_horas == 72

def test_encolar_orden_verde_con_menos_de_5_dias_sla_24h_alta(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     dias_hasta_fin_medicamento=3)
    assert s.prioridad == PrioridadHITL.ALTA
    assert s.sla_horas == 24

def test_sla_acortado_aplica_transversalmente(cola):
    ctx_amarilla_urgente = _crear_ctx_mock("PA_URGENTE")
    s_urgente = cola.encolar(ctx_amarilla_urgente, TipoSolicitudHITL.REVISION_EXAMENES,
                             dias_hasta_fin_medicamento=3)
    assert s_urgente.prioridad == PrioridadHITL.ALTA
    assert s_urgente.sla_horas == 24

    ctx_amarilla_normal = _crear_ctx_mock("PA_NORMAL")
    s_normal = cola.encolar(ctx_amarilla_normal, TipoSolicitudHITL.REVISION_EXAMENES,
                            dias_hasta_fin_medicamento=10)
    assert s_normal.prioridad == PrioridadHITL.NORMAL
    assert s_normal.sla_horas == 72


def test_encolar_marca_ctx_requiere_hitl(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.REVISION_EXAMENES)
    assert ctx.requiere_hitl is True
    assert ctx.hitl_solicitud_id == s.id
    assert len(ctx.auditoria) == 1

def test_obtener_retorna_none_si_no_existe(cola):
    assert cola.obtener("hitl_inexistente") is None

def test_obtener_pendientes_filtra_por_especialista(cola):
    ctx_a = _crear_ctx_mock("PA")
    ctx_b = _crear_ctx_mock("PB")
    cola.encolar(ctx_a, TipoSolicitudHITL.ORDEN_VERDE,
                 especialista_titular="doc_001")
    cola.encolar(ctx_b, TipoSolicitudHITL.REVISION_EXAMENES,
                 especialista_titular="doc_002")
    resultado = cola.obtener_pendientes(especialista_id="doc_001")
    assert len(resultado) == 1
    assert resultado[0].paciente_id == "PA"

def test_obtener_pendientes_filtra_por_prioridad(cola):
    ctx_n = _crear_ctx_mock("PN")
    ctx_a = _crear_ctx_mock("PA")
    cola.encolar(ctx_n, TipoSolicitudHITL.ORDEN_VERDE,
                 dias_hasta_fin_medicamento=10)
    cola.encolar(ctx_a, TipoSolicitudHITL.ORDEN_VERDE,
                 dias_hasta_fin_medicamento=2)
    resultado = cola.obtener_pendientes(prioridad_minima=PrioridadHITL.ALTA)
    assert len(resultado) == 1
    assert resultado[0].prioridad == PrioridadHITL.ALTA

def test_obtener_pendientes_ordena_por_prioridad_y_antiguedad(cola):
    ctx1 = _crear_ctx_mock("P1")
    ctx2 = _crear_ctx_mock("P2")
    ctx3 = _crear_ctx_mock("P3")
    s_normal = cola.encolar(ctx1, TipoSolicitudHITL.ORDEN_VERDE,
                             dias_hasta_fin_medicamento=10)
    s_alta   = cola.encolar(ctx2, TipoSolicitudHITL.ORDEN_VERDE,
                             dias_hasta_fin_medicamento=2)
    s_normal2 = cola.encolar(ctx3, TipoSolicitudHITL.REVISION_EXAMENES)

    pendientes = cola.obtener_pendientes()
    assert pendientes[0].id == s_alta.id
    assert pendientes[1].id == s_normal.id
    assert pendientes[2].id == s_normal2.id

def test_tomar_en_revision_cambia_estado(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.tomar_en_revision(s.id, "doc_001")
    assert cola.obtener(s.id).estado == EstadoHITL.EN_REVISION

def test_tomar_en_revision_bloquea_si_ya_en_revision(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.tomar_en_revision(s.id, "doc_001")
    with pytest.raises(ErrorHITL, match="EN_REVISION"):
        cola.tomar_en_revision(s.id, "doc_002")

def test_aprobar_cambia_estado_y_registra_medico(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK continuar")
    s_final = cola.obtener(s.id)
    assert s_final.estado == EstadoHITL.APROBADA
    assert s_final.especialista_aprobador == "doc_001"
    assert s_final.comentario_medico == "OK continuar"

def test_aprobar_calcula_fecha_aprobacion(cola):
    ctx = _crear_ctx_mock()
    antes = datetime.now()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK")
    despues = datetime.now()
    fecha = cola.obtener(s.id).fecha_aprobacion
    assert antes <= fecha <= despues

def test_no_se_puede_aprobar_solicitud_ya_aprobada(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK")
    with pytest.raises(ErrorHITL, match="aprobada"):
        cola.aprobar(s.id, "doc_001", "segunda vez")

def test_no_se_puede_aprobar_solicitud_rechazada(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.rechazar(s.id, "doc_001", "No aplica")
    with pytest.raises(ErrorHITL, match="rechazada"):
        cola.aprobar(s.id, "doc_001", "intento aprobar rechazada")

def test_rechazar_cambia_estado_con_comentario(cola):
    ctx = _crear_ctx_mock()
    s = cola.encolar(ctx, TipoSolicitudHITL.REVISION_EXAMENES)
    cola.rechazar(s.id, "doc_002", "Faltan examenes complementarios")
    s_final = cola.obtener(s.id)
    assert s_final.estado == EstadoHITL.RECHAZADA
    assert s_final.comentario_medico == "Faltan examenes complementarios"
    assert s_final.fecha_rechazo is not None

def test_estadisticas_retorna_conteos_correctos(cola):
    for pid in ["P1", "P2", "P3"]:
        cola.encolar(_crear_ctx_mock(pid), TipoSolicitudHITL.ORDEN_VERDE)
    ids = [s.id for s in cola.obtener_pendientes()]
    cola.aprobar(ids[0], "doc_001", "OK")
    cola.rechazar(ids[1], "doc_001", "No OK")

    st = cola.estadisticas()
    assert st["total_solicitudes"] == 3
    assert st["por_estado"]["aprobada"]  == 1
    assert st["por_estado"]["rechazada"] == 1
    assert st["por_estado"]["pendiente"] == 1

def test_tasa_aprobacion_pct_calculo_correcto(cola):
    for pid in ["P1", "P2", "P3"]:
        cola.encolar(_crear_ctx_mock(pid), TipoSolicitudHITL.ORDEN_VERDE)
    ids = [s.id for s in cola.obtener_pendientes()]
    cola.aprobar(ids[0], "doc_001", "OK")
    cola.aprobar(ids[1], "doc_001", "OK")
    cola.rechazar(ids[2], "doc_001", "No OK")

    st = cola.estadisticas()
    assert st["tasa_aprobacion_pct"] == pytest.approx(66.7, abs=0.1)

@pytest.fixture
def cola_y_motor():
    cola  = ColaRevisionHITL()
    motor = MotorEscalamientoHITL(cola)
    return cola, motor

def test_motor_escalamiento_no_dispara_antes_de_48h(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("P1")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 10))
    assert eventos == []

def test_motor_escalamiento_fase_a_a_las_48h_envia_alerta(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("P1")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 49))
    assert len(eventos) == 1
    assert eventos[0].fase_protocolo == "A_preventiva"
    assert eventos[0].resultado == "alerta_enviada"
    assert eventos[0].especialista_notificado == "doc_001"

def test_motor_escalamiento_fase_a_no_cambia_estado_solicitud(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("P1")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 49))
    assert cola.obtener(s.id).estado == EstadoHITL.PENDIENTE

def test_motor_escalamiento_fase_a_solo_se_ejecuta_una_vez(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("P1")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 49))
    eventos2 = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 50))
    assert not any(e.fase_protocolo == "A_preventiva" for e in eventos2)

def test_motor_escalamiento_fase_b_a_las_72h_extiende_sla(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PZ")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 49))
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    assert len(eventos) == 1
    assert eventos[0].fase_protocolo == "B_pool"
    assert eventos[0].resultado == "sla_extendido_24h"

def test_motor_escalamiento_fase_b_mantiene_estado_pendiente(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PB")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    assert cola.obtener(s.id).estado == EstadoHITL.PENDIENTE

def test_motor_escalamiento_fase_b_actualiza_prioridad_a_alta(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PP")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    assert cola.obtener(s.id).prioridad == PrioridadHITL.ALTA

def test_motor_escalamiento_fase_b_extiende_sla_24h_adicionales(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PE")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    assert cola.obtener(s.id).sla_horas == 97

def test_motor_escalamiento_fase_c_a_las_96h_emite_receta_emergencia(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PC")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 97))
    ev_c = [e for e in eventos if e.fase_protocolo == "C_emergencia"]
    assert len(ev_c) == 1
    assert ev_c[0].resultado == "fallback_emergencia_activado"
    assert cola.obtener(s.id).datos_contexto["receta_emergencia_emitida"] is True
    assert cola.obtener(s.id).datos_contexto["dias_receta_emergencia"] == 14

def test_motor_escalamiento_fase_c_marca_reclasificacion_amarilla(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PCa")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 73))
    motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 97))
    assert cola.obtener(s.id).datos_contexto["reclasificado_via_amarilla"] is True

def test_motor_escalamiento_fase_c_no_se_ejecuta_si_ya_aprobada(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PY")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK")
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 97))
    assert len(eventos) == 0

def test_motor_escalamiento_secuencia_completa_a_b_c(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PX")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    t0 = s.creada_en

    ev_a = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 49))
    assert len(ev_a) == 1 and ev_a[0].fase_protocolo == "A_preventiva"

    ev_b = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 73))
    assert len(ev_b) == 1 and ev_b[0].resultado == "sla_extendido_24h"
    assert s.esta_activa

    ev_c = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 97))
    assert len(ev_c) == 1 and ev_c[0].fase_protocolo == "C_emergencia"
    assert s.estado == EstadoHITL.EXPIRADA_EMERGENCIA

    st = motor.estadisticas_escalamientos()
    assert st["total_escalamientos"] == 3

def test_motor_escalamiento_solicitud_aprobada_no_escala(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PA")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK")
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 49))
    assert len(eventos) == 0

def test_rechazo_marca_reclasificacion_via_amarilla(cola):
    ctx = _crear_ctx_mock("PR")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.rechazar(s.id, "doc_001", "Necesita control presencial")
    s_final = cola.obtener(s.id)
    assert s_final.datos_contexto["requiere_reclasificacion_via_amarilla"] is True
    assert s_final.datos_contexto["motivo_rechazo"] == "Necesita control presencial"

def test_estadisticas_escalamientos_retorna_conteos_correctos(cola_y_motor):
    cola, motor = cola_y_motor
    ctx1 = _crear_ctx_mock("P1")
    s1 = cola.encolar(ctx1, TipoSolicitudHITL.ORDEN_VERDE,
                      especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s1.creada_en, 49))
    motor.procesar_escalamientos(ahora=_avanzar_horas(s1.creada_en, 73))
    cola.aprobar(s1.id, "doc_001", "Aprobada")

    ctx2 = _crear_ctx_mock("P2")
    s2 = cola.encolar(ctx2, TipoSolicitudHITL.ORDEN_VERDE,
                      especialista_titular="doc_001")
    motor.procesar_escalamientos(ahora=_avanzar_horas(s2.creada_en, 73))
    motor.procesar_escalamientos(ahora=_avanzar_horas(s2.creada_en, 97))

    st = motor.estadisticas_escalamientos()
    assert st["por_fase"]["A_preventiva"] == 1
    assert st["por_fase"]["B_pool"]       == 2
    assert st["por_fase"]["C_emergencia"] == 1
    assert st["solicitudes_que_llegaron_a_fase_c"] == 1
    assert st["tasa_escalamiento_b_exitosa"] == 100.0

def test_solicitudes_aprobadas_durante_periodo_no_escalan(cola_y_motor):
    cola, motor = cola_y_motor
    ctx_a = _crear_ctx_mock("PA")
    ctx_b = _crear_ctx_mock("PB")
    sa = cola.encolar(ctx_a, TipoSolicitudHITL.ORDEN_VERDE)
    sb = cola.encolar(ctx_b, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(sa.id, "doc_001", "OK")
    eventos = motor.procesar_escalamientos(ahora=_avanzar_horas(sb.creada_en, 49))
    assert len(eventos) == 1
    assert cola.obtener(eventos[0].solicitud_id).paciente_id == "PB"

def test_fase_c_se_dispara_despues_de_b_si_nadie_aprobo(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PG1")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    t0 = s.creada_en

    ev_b = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 73))
    assert ev_b[0].resultado == "sla_extendido_24h"
    assert s.estado == EstadoHITL.PENDIENTE

    ev_c = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 97))
    assert len(ev_c) == 1
    assert ev_c[0].fase_protocolo == "C_emergencia"
    assert s.estado == EstadoHITL.EXPIRADA_EMERGENCIA
    assert s.datos_contexto["receta_emergencia_emitida"] is True
    assert s.datos_contexto["dias_receta_emergencia"] == 14

def test_fase_c_NO_se_dispara_si_aprobada_antes_de_96h(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PG2")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    t0 = s.creada_en

    motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 73))
    assert s.estado == EstadoHITL.PENDIENTE

    cola.aprobar(s.id, "doc_001", "Aprobada")
    assert s.estado == EstadoHITL.APROBADA

    ev = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 97))
    assert len(ev) == 0


def test_estado_expirada_emergencia_no_escala_nuevamente(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PG3")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    t0 = s.creada_en

    motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 73))
    motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 97))
    assert s.estado == EstadoHITL.EXPIRADA_EMERGENCIA

    ev = motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 200))
    assert len(ev) == 0

def test_estado_aprobada_nunca_escala(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PG4")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.aprobar(s.id, "doc_001", "OK")
    ev = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 100))
    assert len(ev) == 0


def test_estado_rechazada_nunca_escala(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PG5")
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE)
    cola.rechazar(s.id, "doc_001", "No procede")
    ev = motor.procesar_escalamientos(ahora=_avanzar_horas(s.creada_en, 100))
    assert len(ev) == 0


def test_escalamiento_fase_c_reclasifica_via_amarilla(cola_y_motor):
    cola, motor = cola_y_motor
    ctx = _crear_ctx_mock("PC_RECLASIF", via=ViaRuteo.ROJA)
    s = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                     especialista_titular="doc_001")
    t0 = s.creada_en

    motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 73))
    motor.procesar_escalamientos(ahora=_avanzar_horas(t0, 97))

    assert ctx.via_ruteo == ViaRuteo.AMARILLA
    assert ctx.control_rutina_adelantado is True
    acciones = [d.accion for d in ctx.auditoria]
    assert "reclasificacion_via_amarilla_hitl_c" in acciones


def test_aprobar_registra_entrada_en_auditoria_del_paciente(cola):
    ctx = _crear_ctx_mock("P_AUDIT")
    s   = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE, especialista_titular="doc_001")

    entradas_previas = len(ctx.auditoria)

    cola.aprobar(s.id, "doc_medico_01", "Aprobado sin observaciones")

    assert len(ctx.auditoria) == entradas_previas + 1

    entrada = ctx.auditoria[-1]
    assert entrada.accion   == "hitl_aprobada"
    assert entrada.motor    == "hitl_revision"
    assert entrada.datos_usados["solicitud_id"]   == s.id
    assert entrada.datos_usados["usuario_id"]     == "doc_medico_01"
    assert entrada.datos_usados["estado_final"]   == EstadoHITL.APROBADA.value
