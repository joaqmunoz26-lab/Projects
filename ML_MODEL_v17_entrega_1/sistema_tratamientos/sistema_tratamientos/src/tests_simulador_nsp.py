
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_sim12 = importlib.import_module("12_simulador_nsp")

from servicios.predictor_nsp import (
    COSTO_LLAMADA_PERSONALIZADA_CLP,
    COSTO_SMS_FALLBACK_CLP,
    COSTO_WHATSAPP_CLP,
    REDUCCION_ESCENARIO_A_DOBLE,
    REDUCCION_ESCENARIO_B_LLAMADA,
    REDUCCION_ESCENARIO_C_DIGITAL,
    UMBRAL_RIESGO_ALTO,
    UMBRAL_RIESGO_BAJO,
    PredictorNSP,
    ScoreNSP,
)


def _paciente_bajo_riesgo() -> dict:
    return {
        "paciente_id": "P001",
        "historial_inasistencias": 0,
        "via_clinica": "AMARILLA",
        "tramo_fonasa": "D",
        "hba1c_actual": 7.5,
        "modalidad": "telemedicina",
        "edad": 50,
        "acceso_internet": True,
        "dias_sin_registro": 3,
        "zona_geografica": "URBANO",
    }

def _paciente_critico() -> dict:
    return {
        "paciente_id": "P999",
        "historial_inasistencias": 3,
        "via_clinica": "VERDE",
        "tramo_fonasa": "A",
        "hba1c_actual": 6.5,
        "modalidad": "presencial",
        "edad": 75,
        "acceso_internet": False,
        "dias_sin_registro": 35,
        "zona_geografica": "RURAL_AISLADO",
    }

def test_score_paciente_bajo_riesgo():
    p = PredictorNSP(seed=42)
    score = p.calcular_score(_paciente_bajo_riesgo())

    assert score.categoria == "BAJO", f"Esperado BAJO, obtuvo {score.categoria}"
    assert score.score_total < UMBRAL_RIESGO_BAJO

def test_score_paciente_critico():
    p = PredictorNSP(seed=42)
    score = p.calcular_score(_paciente_critico())

    assert score.categoria == "CRITICO", f"Esperado CRITICO, obtuvo {score.categoria}"
    assert score.score_total >= UMBRAL_RIESGO_ALTO

def test_paciente_via_roja_menor_score_que_verde():
    p    = PredictorNSP(seed=42)
    base = _paciente_bajo_riesgo()

    base["via_clinica"] = "VERDE"
    score_verde = p.calcular_score(base)

    base["via_clinica"] = "ROJA"
    score_roja = p.calcular_score(base)

    assert score_roja.score_total < score_verde.score_total, (
        f"Roja ({score_roja.score_total}) debería ser < Verde ({score_verde.score_total})"
    )

def test_decision_critico_aplica_escenario_A():
    p     = PredictorNSP(seed=42)
    score = ScoreNSP("P1", 85, "CRITICO", {})

    esc, canal, costo, reduccion = p.decidir_intervencion(score, "URBANO")

    assert esc == "A_doble"
    assert "llamada" in canal
    assert reduccion == REDUCCION_ESCENARIO_A_DOBLE
    assert costo >= COSTO_LLAMADA_PERSONALIZADA_CLP

def test_decision_alto_via_verde_aplica_escenario_B():
    p     = PredictorNSP(seed=42)
    score = ScoreNSP("P2", 60, "ALTO", {})

    esc, canal, costo, reduccion = p.decidir_intervencion(score, "URBANO",
                                                          via_clinica="VERDE")

    assert esc == "B_llamada"
    assert canal == "llamada"
    assert costo == COSTO_LLAMADA_PERSONALIZADA_CLP
    assert reduccion == REDUCCION_ESCENARIO_B_LLAMADA

def test_decision_alto_via_amarilla_degrada_a_C():
    p     = PredictorNSP(seed=42)
    score = ScoreNSP("P2b", 60, "ALTO", {})

    esc, canal, costo, reduccion = p.decidir_intervencion(score, "URBANO",
                                                          via_clinica="AMARILLA")

    assert esc == "C_digital"
    assert canal in ("whatsapp", "sms")
    assert costo in (COSTO_WHATSAPP_CLP, COSTO_SMS_FALLBACK_CLP)
    assert reduccion == REDUCCION_ESCENARIO_C_DIGITAL

def test_decision_medio_aplica_escenario_C():
    p     = PredictorNSP(seed=42)
    score = ScoreNSP("P3", 35, "MEDIO", {})

    esc, canal, costo, reduccion = p.decidir_intervencion(score, "URBANO")

    assert esc == "C_digital"
    assert canal in ("whatsapp", "sms")
    assert costo in (COSTO_WHATSAPP_CLP, COSTO_SMS_FALLBACK_CLP)
    assert reduccion == REDUCCION_ESCENARIO_C_DIGITAL

def test_decision_bajo_sin_intervencion():
    p     = PredictorNSP(seed=42)
    score = ScoreNSP("P4", 15, "BAJO", {})

    esc, canal, costo, reduccion = p.decidir_intervencion(score, "URBANO")

    assert esc == "ninguna"
    assert canal == "ninguno"
    assert costo == 0
    assert reduccion == 0.0

def test_constantes_calibradas_literatura_verificada():
    assert REDUCCION_ESCENARIO_B_LLAMADA == 0.49, "Dunstan 2023: 49%"
    assert REDUCCION_ESCENARIO_A_DOBLE   == 0.52, "Compuesto Dunstan+Robotham: 52%"
    assert REDUCCION_ESCENARIO_C_DIGITAL == 0.27, "Hasvold/Robotham: 27%"

def test_score_desglose_tiene_8_variables():
    p = PredictorNSP(seed=42)
    score = p.calcular_score(_paciente_bajo_riesgo())

    variables_requeridas = {
        "historial_inasistencias",
        "via_clinica",
        "tramo_fonasa",
        "hba1c_controlado",
        "barrera_fisica",
        "edad",
        "sin_internet",
        "inercia_paciente",
    }
    assert variables_requeridas == set(score.desglose_componentes.keys()), (
        f"Desglose incompleto: {set(score.desglose_componentes.keys())}"
    )

def test_smoke_simulador_nsp():
    archivo_base = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "simulacion_solo_ews_eventos_completos.csv"
    )
    if not archivo_base.exists():
        pytest.skip("CSV base no disponible; correr operativo longitudinal primero")

    _sim = importlib.import_module("12_simulador_nsp")
    metricas = _sim.simular_capa_nsp(seed=42)

    for clave in [
        "costo_total_clp",
        "costo_intervenciones_clp",
        "costo_rebotes_nsp_clp",
        "n_eventos_control_evaluados",
        "n_intervenciones_A_doble",
        "n_intervenciones_B_llamada",
        "n_intervenciones_C_digital",
        "n_sin_intervencion",
        "n_rebotes_nsp",
        "ahorro_vs_baseline_pct",
        "ahorro_vs_solo_ews_pct",
    ]:
        assert clave in metricas, f"Clave ausente: {clave}"

    assert metricas["costo_total_clp"] > 0
    n = metricas["n_eventos_control_evaluados"]
    assert n > 0
    assert (
        metricas["n_intervenciones_A_doble"]
        + metricas["n_intervenciones_B_llamada"]
        + metricas["n_intervenciones_C_digital"]
        + metricas["n_sin_intervencion"]
    ) == n

    assert metricas["ahorro_vs_solo_ews_pct"] > 0, (
        f"ews_nsp debe ser más barato que solo_ews; obtuvo {metricas['ahorro_vs_solo_ews_pct']:.2f}%"
    )

def test_smoke_escenario_solo_nsp(tmp_path):
    from unittest.mock import patch

    import pandas as pd

    _sim10 = importlib.import_module("10_simulador_eventos")
    params    = _sim10.cargar_parametros("asumido_literatura")
    pacs, log = _sim10.correr_simulacion("base", 20, 400, params, 42)
    with patch.object(_sim10, "RAIZ", tmp_path):
        _sim10.generar_reportes(pacs, log, "base")

    _sim12 = importlib.import_module("12_simulador_nsp")
    rep = tmp_path / "reports"
    with patch.object(_sim12, "DIR_REPORTS", rep):
        metricas = _sim12.simular_capa_nsp_base(seed=42)

    csv = rep / "simulacion_solo_nsp_resumen.csv"
    assert csv.exists(), "simulacion_solo_nsp_resumen.csv no fue creado"

    df = pd.read_csv(csv)
    for col in ("escenario", "n_pacientes", "costo_total_clp"):
        assert col in df.columns, f"Columna ausente: {col}"
    assert df["escenario"].iloc[0]             == "solo_nsp"
    assert metricas["escenario"]               == "solo_nsp"
    assert metricas["n_eventos_control_evaluados"] > 0

def test_smoke_escenario_ews_nsp(tmp_path):
    import pandas as pd

    from core.decisiones import BarreraActivada, ViaRuteo
    from core.fases import FaseActual

    _sim10 = importlib.import_module("10_simulador_eventos")
    _sim12 = importlib.import_module("12_simulador_nsp")

    class _MockEWSAmarilla:
        def __init__(self, cola_hitl=None, **_):
            pass
        def procesar(self, ctx, **_):
            ctx.fase_actual      = FaseActual.FASE_2_EWS
            ctx.via_ruteo        = ViaRuteo.AMARILLA
            ctx.barrera_activada = BarreraActivada.MODELO_ML
            ctx.riesgo_ml        = 0.45
            ctx.modelo_usado     = "mock"
            return ctx

    rep = tmp_path / "reports"
    rep.mkdir()

    with patch("pipeline.fase2_ews.EWSPipeline", _MockEWSAmarilla):
        with patch("core.evolucion_clinica.ModeloEvolucionClinica") as mock_ev:
            mock_ev.return_value.evolucionar_paciente   = lambda **kw: kw.get("datos_clinicos_previos", {})
            mock_ev.return_value.estadisticas_evolucion = lambda: {}
            with patch.object(_sim10, "DIR_REPORTS", rep):
                _sim10.simular_escenario_solo_ews(
                    n_pacientes=20, dias_simulacion=100, seed=42,
                    generar_eventos_completos=True,
                )

    with patch.object(_sim12, "DIR_REPORTS", rep):
        metricas = _sim12.simular_capa_nsp(seed=42)

    csv = rep / "simulacion_ews_nsp_resumen.csv"
    assert csv.exists(), "simulacion_ews_nsp_resumen.csv no fue creado"
    df = pd.read_csv(csv)
    for col in ("escenario", "costo_total_clp"):
        assert col in df.columns, f"Columna ausente: {col}"
    assert df["escenario"].iloc[0] == "ews_nsp"
    assert metricas["escenario"]   == "ews_nsp"
    assert metricas["n_eventos_control_evaluados"] > 0

    intervenciones = (metricas["n_intervenciones_A_doble"]
                      + metricas["n_intervenciones_B_llamada"]
                      + metricas["n_intervenciones_C_digital"])
    assert intervenciones > 0, "NSP no genero ninguna intervencion (todos BAJO)"

def test_smoke_replicas_multiples_nsp(tmp_path):
    import pandas as pd
    _sim10 = importlib.import_module("10_simulador_eventos")
    rep    = tmp_path / "reports"
    rep.mkdir()

    costos = []
    for seed in (100, 101):
        suf    = f"_seed{seed}"
        params = _sim10.cargar_parametros("asumido_literatura")
        pacs, log = _sim10.correr_simulacion("base", 20, 400, params, seed)
        with patch.object(_sim10, "RAIZ", tmp_path):
            _sim10.generar_reportes(pacs, log, "base", seed_sufijo=suf)

        with patch.object(_sim12, "DIR_REPORTS", rep):
            m = _sim12.simular_capa_nsp_base(seed=seed, sufijo=suf)
        costos.append(m["costo_total_clp"])

        csv_out = rep / f"simulacion_solo_nsp_resumen{suf}.csv"
        assert csv_out.exists(), f"{csv_out.name} no fue creado"
        df = pd.read_csv(csv_out)
        assert df["escenario"].iloc[0] == "solo_nsp"

    assert not (rep / "simulacion_solo_nsp_resumen.csv").exists()
    assert len(set(costos)) > 1, f"Replicas con costos identicos: {costos}"
