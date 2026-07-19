import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ev = importlib.import_module("11_evaluador_economico")

cargar_resultados_simulaciones   = _ev.cargar_resultados_simulaciones
construir_tabla_comparativa      = _ev.construir_tabla_comparativa
estimar_costo_ews                = _ev.estimar_costo_ews
analizar_sensibilidad_parametros = _ev.analizar_sensibilidad_parametros
generar_tabla_markdown           = _ev.generar_tabla_markdown
generar_reporte_completo_tesis   = _ev.generar_reporte_completo_tesis

def _resumen_base():
    return {
        "escenario": "base", "n_pacientes": 500,
        "total_eventos": 1767, "total_presenciales": 1767,
        "total_telemedicina": 0, "total_asincrona": 0,
        "total_sin_consulta": 0, "costo_total_clp": 57_126_000,
        "tiempo_total_min": 145_620, "rebotes_totales": 243,
    }

def _resumen_solo_ews():
    return {
        "total_procesados": 500, "fase1_confirmados": 441,
        "fase1_no_confirmados": 59, "fase2_via_verde": 148,
        "fase2_via_amarilla": 203, "fase2_via_roja": 90,
        "barrera_duras_bypass": 18, "uso_fallback_red": 0,
        "fallback_fallo_red_saturada": 0, "hitl_encolado": 148,
        "hitl_aprobado_normal": 0, "hitl_escalado_t48": 3,
        "hitl_escalado_t72": 3, "hitl_escalado_t96_emergencia": 148,
        "control_rutina_adelantado": 148,
        "aprobadas_por_humano": 135, "rechazadas_por_humano": 6,
        "aprobadas_en_sla_normal": 100, "aprobadas_en_fase_a": 25,
        "aprobadas_en_fase_b": 10,
        "costo_eventos_total_clp": 27_528_000,
        "tiempo_paciente_total_min": 28_000,
        "eventos_completos_total": 1_800,
        "rebotes_generados": 79, "tasa_rebote_pct": 4.4,
        "transiciones_totales": 64,
        "pacientes_con_via_verde_alguna_vez": 175,
        "pacientes_que_mejoraron_de_via": 59,
        "pacientes_que_empeoraron_de_via": 5,
    }

def _resumen_solo_nsp():
    return {
        "escenario": "solo_nsp", "n_pacientes": 500,
        "costo_total_clp": 45_200_000,
        "costo_eventos_sin_rebote_clp": 44_800_000,
        "costo_intervenciones_clp": 120_000,
        "costo_rebotes_nsp_clp": 280_000,
        "n_eventos_control_evaluados": 1_500,
        "n_intervenciones_A_doble": 5,
        "n_intervenciones_B_llamada": 80,
        "n_intervenciones_C_digital": 400,
        "n_sin_intervencion": 1015,
        "n_rebotes_nsp": 10,
        "ahorro_vs_baseline_pct": 6.5,
    }

def _resumen_ews_nsp():
    return {
        "escenario": "ews_nsp",
        "costo_total_clp": 27_100_000,
        "costo_eventos_sin_rebote_clp": 26_600_000,
        "costo_intervenciones_clp": 350_000,
        "costo_rebotes_nsp_clp": 150_000,
        "n_eventos_control_evaluados": 964,
        "n_intervenciones_A_doble": 3,
        "n_intervenciones_B_llamada": 153,
        "n_intervenciones_C_digital": 572,
        "n_sin_intervencion": 236,
        "n_rebotes_nsp": 5,
        "ahorro_vs_baseline_pct": 43.97,
        "ahorro_vs_solo_ews_pct": 1.6,
    }

def _mock_resultados():
    return {
        "Base (100% presencial)": {"resumen": _resumen_base(),     "pacientes": pd.DataFrame(), "archivo_origen": "base"},
        "Solo EWS (sin NSP)":     {"resumen": _resumen_solo_ews(), "pacientes": pd.DataFrame(), "archivo_origen": "solo_ews"},
        "Solo NSP (sin EWS)":     {"resumen": _resumen_solo_nsp(), "pacientes": pd.DataFrame(), "archivo_origen": "solo_nsp"},
        "EWS + NSP":              {"resumen": _resumen_ews_nsp(),  "pacientes": pd.DataFrame(), "archivo_origen": "ews_nsp"},
    }

def test_cargar_resultados_si_existen_simulaciones(tmp_path):
    for nombre, resumen in [
        ("base",     _resumen_base()),
        ("solo_ews", _resumen_solo_ews()),
        ("solo_nsp", _resumen_solo_nsp()),
        ("ews_nsp",  _resumen_ews_nsp()),
    ]:
        pd.DataFrame([resumen]).to_csv(
            tmp_path / f"simulacion_{nombre}_resumen.csv", index=False
        )

    resultados = cargar_resultados_simulaciones(tmp_path)

    assert len(resultados) == 4
    assert "Base (100% presencial)" in resultados
    assert "Solo EWS (sin NSP)"     in resultados
    assert "Solo NSP (sin EWS)"     in resultados
    assert "EWS + NSP"              in resultados


def test_construir_tabla_comparativa_normaliza_columnas():
    df = construir_tabla_comparativa(_mock_resultados())

    assert len(df) == 4
    for col in ["costo_total_clp", "costo_por_confirmado"]:
        assert col in df.columns, f"Columna '{col}' ausente"

    ews = df[df["escenario"] == "Solo EWS (sin NSP)"].iloc[0]
    assert ews["dm2_confirmados"] == 441
    assert ews["costo_por_confirmado"] > 0

    nsp = df[df["escenario"] == "Solo NSP (sin EWS)"].iloc[0]
    assert nsp["costo_total_clp"] == 45_200_000


def test_estimar_costo_ews_calcula_distribucion_correcta():
    assert estimar_costo_ews({"fase2_via_roja": 10,
                               "fase2_via_amarilla": 0,
                               "fase2_via_verde": 0}) == 280_000

    assert estimar_costo_ews({"fase2_via_verde": 10,
                               "fase2_via_amarilla": 0,
                               "fase2_via_roja": 0}) == 60_000

    costo = estimar_costo_ews({"fase2_via_amarilla": 10,
                                "fase2_via_verde": 0,
                                "fase2_via_roja": 0})
    ESPERADO_AMARILLA = 150_400
    assert costo == ESPERADO_AMARILLA, f"Esperado {ESPERADO_AMARILLA}, obtuvo {costo}"

def test_analisis_sensibilidad_genera_9_combinaciones():
    df   = construir_tabla_comparativa(_mock_resultados())
    df_s = analizar_sensibilidad_parametros(df)

    assert len(df_s) == 9, f"Esperado 9 filas, obtuvo {len(df_s)}"
    assert set(df_s["parametro_variado"]) == {
        "tasa_rebote", "costo_presencial", "pct_rural"
    }
    for param in ["tasa_rebote", "costo_presencial", "pct_rural"]:
        assert len(df_s[df_s["parametro_variado"] == param]) == 3

def test_generar_tabla_markdown_incluye_4_escenarios():
    df = construir_tabla_comparativa(_mock_resultados())
    md = generar_tabla_markdown(df)

    for nombre in ["Base (100% presencial)", "Solo EWS (sin NSP)",
                   "Solo NSP (sin EWS)", "EWS + NSP"]:
        assert nombre in md, f"Escenario '{nombre}' ausente en tabla MD"

def test_reporte_completo_tesis_estructura_completa(tmp_path):
    df   = construir_tabla_comparativa(_mock_resultados())
    df_s = analizar_sensibilidad_parametros(df)
    ruta = tmp_path / "reporte_test.md"

    generar_reporte_completo_tesis(df, df_s, ruta)

    assert ruta.exists(), "El reporte no fue generado"
    contenido = ruta.read_text(encoding="utf-8")

    for i in range(1, 4):
        assert f"## {i}." in contenido, f"Seccion '## {i}.' ausente en reporte"

def test_evaluador_agregacion_replicas(tmp_path):
    _sim10 = importlib.import_module("10_simulador_eventos")
    from unittest.mock import patch

    rep = tmp_path / "reports"
    rep.mkdir()

    params = _sim10.cargar_parametros("asumido_literatura")
    for seed in (100, 101, 102):
        pacs, log = _sim10.correr_simulacion("base", 20, 400, params, seed)
        with patch.object(_sim10, "RAIZ", tmp_path):
            _sim10.generar_reportes(pacs, log, "base", seed_sufijo=f"_seed{seed}")

    dir_fin = tmp_path / "evaluacion_final"
    _ev.ejecutar_evaluacion_replicas(
        dir_reports=rep, dir_final=dir_fin,
        n_replicas=3, seed=100, escenarios=["base"],
    )

    csv_out = dir_fin / "evaluacion_base_agregado.csv"
    assert csv_out.exists(), "evaluacion_base_agregado.csv no fue creado"

    df = pd.read_csv(csv_out)
    for col in ("metrica", "media", "sd", "ic95_inf", "ic95_sup", "n_replicas"):
        assert col in df.columns, f"Columna ausente: {col}"

    assert (df["n_replicas"] == 3).all(), "n_replicas deberia ser 3 en todas las filas"

    fila_costo = df[df["metrica"] == "costo_total_clp"]
    assert len(fila_costo) == 1, "Falta metrica costo_total_clp"
    assert fila_costo["media"].iloc[0] > 0, "Media de costo debe ser positiva"
