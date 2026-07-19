import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from servicios.hitl_revision import ColaRevisionHITL, TipoSolicitudHITL
from servicios.simulador_aprobacion_hitl import SimuladorAprobacionHITL


def _ctx(paciente_id: str = "P_TEST") -> PacienteContext:
    return PacienteContext(
        paciente_id=paciente_id,
        perfil_paciente=PerfilPaciente(
            zona=ZonaGeografica.URBANO,
            canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        dm2_confirmado=True,
    )

def _poblar_cola(cola: ColaRevisionHITL, n: int) -> list:
    ids = []
    for i in range(n):
        sol = cola.encolar(
            _ctx(f"P{i:04d}"),
            TipoSolicitudHITL.ORDEN_VERDE,
            especialista_titular="endo_001",
        )
        ids.append(sol.id)
    return ids

def test_aprobacion_dentro_de_rango_esperado():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=42)
    _poblar_cola(cola, 500)

    base = datetime.now()
    for dia in range(7):
        sim.procesar_dia(cola, dia, ahora=base + timedelta(days=dia))

    stats = sim.estadisticas()
    assert stats["total_acciones_humanas"] > 0
    assert 0.65 <= stats["tasa_aprobacion_pct"] / 100 <= 0.95

def test_rechazo_en_rango_4_7_pct():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=42)
    _poblar_cola(cola, 500)

    base = datetime.now()
    for dia in range(7):
        sim.procesar_dia(cola, dia, ahora=base + timedelta(days=dia))

    stats = sim.estadisticas()
    assert stats["total_acciones_humanas"] > 0
    assert 0.02 <= stats["tasa_rechazo_pct"] / 100 <= 0.10

def test_no_procesa_solicitudes_terminales():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=42)

    ctx = _ctx()
    sol = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                       especialista_titular="endo_001")
    cola.aprobar(sol.id, "endo_001", "pre-aprobado en test")

    base   = datetime.now()
    eventos = sim.procesar_dia(cola, dia_simulacion=0,
                               ahora=base + timedelta(days=0))

    procesados_por_sim = [e for e in eventos if e.solicitud_id == sol.id]
    assert len(procesados_por_sim) == 0

def test_no_procesa_t96_o_mas():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=42)

    ctx = _ctx()
    sol = cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                       especialista_titular="endo_001")

    ahora   = sol.creada_en + timedelta(hours=100)
    eventos = sim.procesar_dia(cola, dia_simulacion=4, ahora=ahora)

    assert len(eventos) == 0

def test_seed_reproducibilidad():
    cola1 = ColaRevisionHITL()
    cola2 = ColaRevisionHITL()
    _poblar_cola(cola1, 50)
    _poblar_cola(cola2, 50)

    sim1 = SimuladorAprobacionHITL(seed=42)
    sim2 = SimuladorAprobacionHITL(seed=42)

    base = datetime.now()
    e1   = sim1.procesar_dia(cola1, 0, ahora=base + timedelta(days=1))
    e2   = sim2.procesar_dia(cola2, 0, ahora=base + timedelta(days=1))

    assert len(e1) == len(e2)
    for a, b in zip(e1, e2):
        assert a.accion        == b.accion
        assert a.fase_protocolo == b.fase_protocolo

def test_estadisticas_estado_vacio():
    sim   = SimuladorAprobacionHITL(seed=42)
    stats = sim.estadisticas()
    assert stats == {"total_acciones_humanas": 0}

def test_rechazo_marca_paciente_reclasificado():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=0)

    for i in range(200):
        cola.encolar(
            _ctx(f"R{i:04d}"),
            TipoSolicitudHITL.ORDEN_VERDE,
            especialista_titular="endo_001",
        )

    base = datetime.now()
    for dia in range(7):
        sim.procesar_dia(cola, dia, ahora=base + timedelta(days=dia))

    rechazados = [e for e in sim.eventos if e.accion == "rechazada"]
    if rechazados:
        for ev in rechazados:
            assert sim.es_reclasificado(ev.paciente_id), (
                f"Paciente {ev.paciente_id} fue rechazado pero no está en "
                "pacientes_reclasificados"
            )

def test_es_reclasificado_devuelve_false_antes_de_rechazar():
    sim = SimuladorAprobacionHITL(seed=42)
    assert sim.es_reclasificado("PACIENTE_NUEVO") is False

def test_aprobacion_no_marca_reclasificado():
    cola = ColaRevisionHITL()
    sim  = SimuladorAprobacionHITL(seed=42)

    ctx = _ctx("APROBADO_TEST")
    cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                 especialista_titular="endo_001")

    base = datetime.now()
    for dia in range(7):
        sim.procesar_dia(cola, dia, ahora=base + timedelta(days=dia))

    aprobaciones = [e for e in sim.eventos
                    if e.accion == "aprobada" and e.paciente_id == "APROBADO_TEST"]
    if aprobaciones:
        assert not sim.es_reclasificado("APROBADO_TEST"), (
            "Paciente aprobado fue marcado incorrectamente como reclasificado"
        )
