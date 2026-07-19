import importlib
import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

_sim = importlib.import_module("10_simulador_eventos")

simular_ews                   = _sim.simular_escenario_solo_ews
generar_eventos_anuales       = _sim.generar_eventos_anuales_por_paciente
simular_longitudinal          = _sim.simular_paciente_longitudinal
actualizar_disponibilidad     = _sim.actualizar_disponibilidad_red
liberar_cupos                 = _sim.liberar_cupos_diarios

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
    MotorEscalamientoHITL,
    TipoSolicitudHITL,
)
from servicios.red_especialistas import RedEspecialistas


class _MockEWSPipeline:

    def __init__(self, cola_hitl=None, **_):
        self._cola = cola_hitl

    def procesar(self, ctx, evento="control_programado",
                 dias_hasta_fin_medicamento=None):
        ctx.fase_actual      = FaseActual.FASE_2_EWS
        ctx.via_ruteo        = ViaRuteo.VERDE
        ctx.barrera_activada = BarreraActivada.MODELO_ML
        ctx.riesgo_ml        = 0.10
        ctx.modelo_usado     = "mock"
        if self._cola:
            self._cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE,
                               especialista_titular=ctx.especialista_titular)
        return ctx

def _correr_ews_mock(n: int = 10, dias: int = 10, tmp_path=None, **kwargs) -> dict:
    target = tmp_path if tmp_path is not None else Path(tempfile.mkdtemp())
    with patch("pipeline.fase2_ews.EWSPipeline", _MockEWSPipeline):
        with patch("core.evolucion_clinica.ModeloEvolucionClinica") as mock_evol:
            mock_evol.return_value.evolucionar_paciente = lambda **kw: kw.get(
                "datos_clinicos_previos", {}
            )
            mock_evol.return_value.estadisticas_evolucion = lambda: {}
            with patch.object(_sim, "DIR_REPORTS", target):
                return simular_ews(n_pacientes=n, dias_simulacion=dias, seed=42, **kwargs)

def test_escenario_ews_simula_n_pacientes_correcto():
    n = 15
    metricas = _correr_ews_mock(n=n, dias=10)
    assert metricas["total_procesados"] == n

def test_escenario_ews_genera_los_5_csvs(tmp_path):
    _correr_ews_mock(n=10, dias=10, tmp_path=tmp_path)

    esperados = [
        "simulacion_solo_ews_resumen.csv",
        "simulacion_solo_ews_eventos.csv",
        "simulacion_solo_ews_decisiones.csv",
        "simulacion_solo_ews_hitl.csv",
        "simulacion_solo_ews_escalamientos.csv",
    ]
    for fname in esperados:
        assert (tmp_path / fname).exists(), f"Falta: {fname}"

def test_metricas_suman_total_procesados():
    metricas = _correr_ews_mock(n=20, dias=20)
    total = metricas["total_procesados"]
    suma  = metricas["fase1_confirmados"] + metricas["fase1_no_confirmados"]
    assert suma == total, f"Suma fases ({suma}) != total ({total})"

def test_metricas_via_solo_para_confirmados():
    metricas   = _correr_ews_mock(n=20, dias=20)
    suma_vias  = (metricas["fase2_via_verde"]
                  + metricas["fase2_via_amarilla"]
                  + metricas["fase2_via_roja"])
    confirmados = metricas["fase1_confirmados"]
    assert suma_vias <= confirmados, (
        f"suma_vias ({suma_vias}) > confirmados ({confirmados})"
    )

def test_disponibilidad_red_se_reduce_domingos():
    red = RedEspecialistas()
    liberar_cupos(red, 0, {})
    actualizar_disponibilidad(red, dia=6, params={})
    disponibles = sum(1 for e in red.listar_todos() if e.disponible)
    assert disponibles == 4, f"Esperado 4 disponibles en domingo, obtuvo {disponibles}"

def test_disponibilidad_red_normal_todos_disponibles():
    red = RedEspecialistas()
    liberar_cupos(red, 0, {})
    actualizar_disponibilidad(red, dia=1, params={})
    disponibles = sum(1 for e in red.listar_todos() if e.disponible)
    assert disponibles == 10

def test_disponibilidad_red_vacaciones_marzo():
    red = RedEspecialistas()
    liberar_cupos(red, 0, {})
    actualizar_disponibilidad(red, dia=70, params={})
    disponibles = sum(1 for e in red.listar_todos() if e.disponible)
    assert disponibles == 8

def test_escalamientos_se_acumulan_correctamente():
    cola  = ColaRevisionHITL()
    motor = MotorEscalamientoHITL(cola)

    ctx = PacienteContext(
        paciente_id="TEST_ESCAL",
        perfil_paciente=PerfilPaciente(
            ZonaGeografica.URBANO,
            CanalDiagnostico.PRESENCIAL_CENTRO,
            CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        dm2_confirmado=True,
    )
    now = datetime.now()
    cola.encolar(ctx, TipoSolicitudHITL.ORDEN_VERDE, especialista_titular="endo_001")

    motor.procesar_escalamientos(ahora=now + timedelta(hours=49))
    motor.procesar_escalamientos(ahora=now + timedelta(hours=73))
    motor.procesar_escalamientos(ahora=now + timedelta(hours=97))

    fases = {e.fase_protocolo for e in motor.eventos}
    assert "A_preventiva" in fases,  "Phase A no se registró"
    assert "B_pool"        in fases,  "Phase B no se registró"
    assert "C_emergencia"  in fases,  "Phase C no se registró"
    assert len(motor.eventos) == 3

def test_generar_eventos_anuales_paciente_confirmado():
    ctx = MagicMock()
    ctx.dm2_confirmado = True
    ctx.paciente_id    = "TEST_VERDE"
    ctx.via_ruteo      = ViaRuteo.VERDE
    ctx.perfil_paciente.canal_monitoreo.value = "app_autonoma"

    eventos = generar_eventos_anuales(ctx, random.Random(42))
    assert len(eventos) == 4, f"Esperado 4 controles, obtuvo {len(eventos)}"

def test_generar_eventos_paciente_no_confirmado():
    ctx = MagicMock()
    ctx.dm2_confirmado = False
    ctx.paciente_id    = "TEST_NOCONF"

    eventos = generar_eventos_anuales(ctx, random.Random(42))
    assert len(eventos) == 1
    assert eventos[0]["tipo_evento"] == "diagnostico_fase1"

def test_via_roja_presencial_genera_costo_fonasa_2025():
    ctx = MagicMock()
    ctx.dm2_confirmado = True
    ctx.paciente_id    = "TEST_ROJA"
    ctx.via_ruteo      = ViaRuteo.ROJA
    ctx.perfil_paciente.zona.value = "urbano"

    eventos = generar_eventos_anuales(ctx, random.Random(0))
    assert len(eventos) == 4
    for e in eventos:
        assert e["costo_consulta_clp"] == 10_380, f"Consulta inesperada: {e['costo_consulta_clp']}"
        assert e["costo_viaje_clp"]    == 2_000,  f"Viaje inesperado: {e['costo_viaje_clp']}"
        assert e["costo_clp"]          == 12_380, f"Total inesperado: {e['costo_clp']}"

def test_via_verde_asincrona_genera_costo_fonasa_2025():
    ctx = MagicMock()
    ctx.dm2_confirmado = True
    ctx.paciente_id    = "TEST_VERDE2"
    ctx.via_ruteo      = ViaRuteo.VERDE

    eventos = generar_eventos_anuales(ctx, random.Random(0))
    assert len(eventos) == 4
    for e in eventos:
        assert e["costo_consulta_clp"] == 2_158, f"Consulta inesperada: {e['costo_consulta_clp']}"
        assert e["costo_viaje_clp"]    == 0,     f"Viaje inesperado (asíncrona no incluye viaje): {e['costo_viaje_clp']}"
        assert e["costo_clp"]          == 2_158, f"Total inesperado: {e['costo_clp']}"

def test_simulacion_ews_genera_csv_eventos_completos(tmp_path):
    metricas = _correr_ews_mock(n=50, dias=30, tmp_path=tmp_path,
                                generar_eventos_completos=True)

    assert "costo_eventos_total_clp"   in metricas, "Falta costo_eventos_total_clp"
    assert "eventos_completos_total"   in metricas, "Falta eventos_completos_total"
    assert metricas["eventos_completos_total"] > 0, "No se generaron eventos"

    csv_path = tmp_path / "simulacion_solo_ews_eventos_completos.csv"
    assert csv_path.exists(), f"CSV no creado: {csv_path.name}"

def test_smoke_escenario_base(tmp_path):
    import pandas as pd

    params    = _sim.cargar_parametros("asumido_literatura")
    pacientes, log_ev = _sim.correr_simulacion("base", 20, 400, params, 42)

    with patch.object(_sim, "RAIZ", tmp_path):
        resumen = _sim.generar_reportes(pacientes, log_ev, "base")

    csv = tmp_path / "reports" / "simulacion_base_resumen.csv"
    assert csv.exists(), "simulacion_base_resumen.csv no fue creado"

    df = pd.read_csv(csv)
    for col in ("escenario", "n_pacientes", "costo_total_clp"):
        assert col in df.columns, f"Columna ausente en resumen: {col}"
    assert df["escenario"].iloc[0]   == "base"
    assert df["n_pacientes"].iloc[0] == 20
    assert resumen["escenario"]          == "base"
    assert resumen["costo_total_clp"]    >= 0
    assert resumen["total_presenciales"] > 0

class _RngMock:
    def __init__(self, valor: float):
        self._valor = valor
    def random(self) -> float:
        return self._valor
    def uniform(self, low, high):
        return (low + high) / 2.0

class _ViaFijaPipeline:
    def __init__(self, via, pool_tipo=None):
        self._via       = via
        self._pool_tipo = pool_tipo
    def procesar(self, ctx, **_):
        ctx.via_ruteo = self._via
        if self._pool_tipo is not None:
            ctx.pool_tipo_asignacion = self._pool_tipo
        return ctx

class _NoopEvolucion:
    def evolucionar_paciente(self, **kw):
        return kw.get("datos_clinicos_previos", {})

def _ctx_verde_urbano() -> PacienteContext:
    ctx = PacienteContext(
        paciente_id="T_VERDE",
        perfil_paciente=PerfilPaciente(
            ZonaGeografica.URBANO,
            CanalDiagnostico.PRESENCIAL_CENTRO,
            CanalMonitoreo.APP_AUTONOMA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        dm2_confirmado=True,
        datos_clinicos={
            "hba1c": 6.5, "glucosa_ayunas": 110.0,
            "adherencia_tratamiento": 0.85,
            "presion_sistolica": 120.0, "presion_diastolica": 78.0,
            "funcion_renal_egfr": 92.0,
        },
    )
    ctx.via_ruteo = ViaRuteo.VERDE
    return ctx

def _ctx_amarilla_presencial_rural_aislado() -> PacienteContext:
    ctx = PacienteContext(
        paciente_id="T_AMARILLA_RA",
        perfil_paciente=PerfilPaciente(
            ZonaGeografica.RURAL_AISLADO,
            CanalDiagnostico.TELECONSULTA_POSTA,
            CanalMonitoreo.TENS_INGRESA,
        ),
        fase_actual=FaseActual.FASE_2_EWS,
        dm2_confirmado=True,
        datos_clinicos={
            "hba1c": 8.0, "glucosa_ayunas": 160.0,
            "adherencia_tratamiento": 0.65,
            "presion_sistolica": 135.0, "presion_diastolica": 88.0,
            "funcion_renal_egfr": 75.0,
        },
    )
    ctx.via_ruteo = ViaRuteo.AMARILLA
    return ctx

def test_inasistencia_amarilla_definitiva_sin_consulta():
    ctx = _ctx_verde_urbano()
    rng_noshow = _RngMock(0.05)

    eventos, _, _ = simular_longitudinal(
        ctx_inicial=ctx,
        pipeline_ews=_ViaFijaPipeline(ViaRuteo.AMARILLA),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=rng_noshow,
        n_controles=1,
    )
    tipos = [e["tipo_evento"] for e in eventos]
    assert "inasistencia_ews" in tipos, f"Falta inasistencia_ews en {tipos}"
    assert not any("control_via" in t for t in tipos), \
        f"Model B: el no-show es definitivo, no debe haber consulta en {tipos}"
    ins = next(e for e in eventos if e["tipo_evento"] == "inasistencia_ews")
    assert ins["costo_clp"] == 8_630 + 500

def test_sin_inasistencia_verde_solo_consulta():
    ctx = _ctx_verde_urbano()
    rng_no_noshow = _RngMock(0.99)

    eventos, _, _ = simular_longitudinal(
        ctx_inicial=ctx,
        pipeline_ews=_ViaFijaPipeline(ViaRuteo.VERDE),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=rng_no_noshow,
        n_controles=1,
    )
    tipos = [e["tipo_evento"] for e in eventos]
    assert "inasistencia_ews" not in tipos
    controles = [e for e in eventos if e["control_num"] >= 1]
    assert len(controles) >= 1
    assert all(c["tipo_evento"].startswith("control_via") for c in controles)

def test_pool_saturado_amarilla_presencial_genera_reprogramacion():
    ctx = _ctx_amarilla_presencial_rural_aislado()
    rng = _RngMock(0.99)

    eventos, _, _ = simular_longitudinal(
        ctx_inicial=ctx,
        pipeline_ews=_ViaFijaPipeline(ViaRuteo.AMARILLA, pool_tipo="saturado"),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=rng,
        n_controles=1,
    )
    tipos = [e["tipo_evento"] for e in eventos]
    assert "reprogramacion_pool_saturado" in tipos, f"Falta reprogramacion en {tipos}"
    reprog = next(e for e in eventos if e["tipo_evento"] == "reprogramacion_pool_saturado")
    assert not any(
        e["tipo_evento"].startswith("control_via") and e["control_num"] == reprog["control_num"]
        for e in eventos
    ), f"El control saturado no debería generar consulta: {tipos}"

def test_pool_saturado_costo_rural_aislado():
    ctx = _ctx_amarilla_presencial_rural_aislado()
    rng = _RngMock(0.99)

    eventos, _, _ = simular_longitudinal(
        ctx_inicial=ctx,
        pipeline_ews=_ViaFijaPipeline(ViaRuteo.AMARILLA, pool_tipo="saturado"),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=rng,
        n_controles=1,
    )
    sat = next(e for e in eventos if e["tipo_evento"] == "reprogramacion_pool_saturado")
    assert sat["costo_clp"] == 60_500, f"Costo inesperado: {sat['costo_clp']}"

def test_pool_saturado_verde_no_genera_reprogramacion():
    ctx = _ctx_verde_urbano()
    rng = _RngMock(0.99)

    eventos, _, _ = simular_longitudinal(
        ctx_inicial=ctx,
        pipeline_ews=_ViaFijaPipeline(ViaRuteo.VERDE, pool_tipo="saturado"),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=rng,
        n_controles=1,
    )
    tipos = [e["tipo_evento"] for e in eventos]
    assert "reprogramacion_pool_saturado" not in tipos
    controles = [e for e in eventos if e["control_num"] >= 1]
    assert len(controles) >= 1

def test_liberar_cupos_resetea_prestamos_diarios():
    from servicios.red_especialistas import RedEspecialistas
    from servicios.red_especialistas import ZonaGeografica as ZG
    red = RedEspecialistas()
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo1, _ = red.asignar_con_cascade(ZG.RURAL_CERCANO)
    assert tipo1 == "prestamo"
    tipo2, _ = red.asignar_con_cascade(ZG.RURAL_CERCANO)
    assert tipo2 == "saturado"
    liberar_cupos(red, 0, {})
    tipo3, eid3 = red.asignar_con_cascade(ZG.RURAL_CERCANO)
    assert tipo3 == "nativo"

def test_reset_pool_entre_pacientes_longitudinales():
    from pipeline.fase3_fallback import (
        agendar_presencial_urgente,
        set_red_especialistas,
    )
    from servicios.red_especialistas import RedEspecialistas

    red = RedEspecialistas()
    set_red_especialistas(red)

    class _RojaPipeline:
        def procesar(self, ctx, **_):
            ctx.via_ruteo        = ViaRuteo.ROJA
            ctx.dm2_confirmado   = True
            ctx.barrera_activada = BarreraActivada.MODELO_ML
            agendar_presencial_urgente(ctx, "doc_001")
            return ctx

    def _ctx_roja_urbano(pid: str) -> PacienteContext:
        ctx = PacienteContext(
            paciente_id=pid,
            perfil_paciente=PerfilPaciente(
                ZonaGeografica.URBANO,
                CanalDiagnostico.PRESENCIAL_CENTRO,
                CanalMonitoreo.APP_AUTONOMA,
            ),
            fase_actual=FaseActual.FASE_2_EWS,
            dm2_confirmado=True,
            datos_clinicos={
                "hba1c": 9.5, "glucosa_ayunas": 200.0,
                "adherencia_tratamiento": 0.40,
                "presion_sistolica": 150.0, "presion_diastolica": 95.0,
                "funcion_renal_egfr": 65.0,
            },
        )
        ctx.via_ruteo = ViaRuteo.ROJA
        return ctx

    for i in range(1, 7):
        red.buscar_por_id(f"doc_{i:03d}").carga_actual_consultas = 1

    ctx_sin_reset = _ctx_roja_urbano("T_SIN_RESET")
    _, vias_sin, _ = simular_longitudinal(
        ctx_inicial=ctx_sin_reset,
        pipeline_ews=_RojaPipeline(),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=_RngMock(0.99),
        n_controles=2,
    )
    pool_control_2_sin_reset = vias_sin[1]["pool_tipo"]
    assert pool_control_2_sin_reset == "saturado", (
        f"Sin reset control 2 debería ser saturado, obtuvo: {pool_control_2_sin_reset}"
    )

    for i in range(1, 7):
        red.buscar_por_id(f"doc_{i:03d}").carga_actual_consultas = 1
    red.resetear_cargas_diarias()

    ctx_con_reset = _ctx_roja_urbano("T_CON_RESET")
    _, vias_con, _ = simular_longitudinal(
        ctx_inicial=ctx_con_reset,
        pipeline_ews=_RojaPipeline(),
        modelo_evolucion=_NoopEvolucion(),
        rng_eventos=_RngMock(0.99),
        n_controles=2,
    )
    pool_control_2_con_reset = vias_con[1]["pool_tipo"]
    assert pool_control_2_con_reset == "nativo", (
        f"Con reset control 2 debería ser nativo, obtuvo: {pool_control_2_con_reset}"
    )

    set_red_especialistas(None)

def test_smoke_replicas_multiples(tmp_path):
    import pandas as pd

    params = _sim.cargar_parametros("asumido_literatura")
    costos = []
    for seed in (100, 101, 102):
        pacs, log = _sim.correr_simulacion("base", 20, 400, params, seed)
        with patch.object(_sim, "RAIZ", tmp_path):
            _sim.generar_reportes(pacs, log, "base", seed_sufijo=f"_seed{seed}")
        csv = tmp_path / "reports" / f"simulacion_base_resumen_seed{seed}.csv"
        assert csv.exists(), f"{csv.name} no fue creado"
        costos.append(int(pd.read_csv(csv)["costo_total_clp"].iloc[0]))

    assert not (tmp_path / "reports" / "simulacion_base_resumen.csv").exists()
    assert len(set(costos)) > 1, f"Todas las replicas dieron el mismo costo: {costos}"
