
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from servicios.predictor_nsp import (
    PredictorNSP,
)

DIR_REPORTS  = Path(__file__).resolve().parent.parent / "reports"
COSTO_REBOTE = 10_380

def _leer_costo_base() -> int:
    ruta = DIR_REPORTS / "simulacion_base_resumen.csv"
    if ruta.exists():
        try:
            return int(pd.read_csv(ruta)["costo_total_clp"].iloc[0])
        except Exception:
            pass
    return 57_126_000

BASELINE_CLP = _leer_costo_base()

P_REBOTE_VIA = {
    "VERDE":    0.00,
    "AMARILLA": 0.10,
    "ROJA":     0.05,
}

_FONASA_PESOS = [162, 147, 120, 71]
_ZONAS        = ["URBANO", "RURAL_CERCANO", "RURAL_AISLADO"]
_ZONAS_PESOS  = [0.85, 0.10, 0.05]

def _perfil_sintetico(paciente_id: str, seed_global: int) -> dict:
    pid_num = int("".join(c for c in paciente_id if c.isdigit()) or 0)
    rng_p   = np.random.default_rng(seed_global + pid_num)

    tramo   = rng_p.choice(["A", "B", "C", "D"], p=[w / 500 for w in _FONASA_PESOS])
    zona    = rng_p.choice(_ZONAS, p=_ZONAS_PESOS)
    edad    = int(np.clip(rng_p.normal(60, 8), 40, 85))
    hba1c   = float(round(np.clip(rng_p.normal(7.4, 0.8), 5.5, 12.0), 1))
    sin_int = rng_p.random() > (0.90 if "URBANO" in zona else 0.70)

    return {
        "tramo_fonasa":    tramo,
        "zona_geografica": zona,
        "edad":            edad,
        "hba1c_actual":    hba1c,
        "acceso_internet": not sin_int,
    }

def _aplicar_nsp_sobre_eventos(
    df_eventos: pd.DataFrame,
    seed: int,
    via_clinica_default: str,
    p_rebote_via_override: dict,
    nombre_escenario: str,
    sufijo: str = "",
) -> dict:
    predictor      = PredictorNSP(seed=seed)
    rng_asistencia = np.random.default_rng(seed + 1000)

    rng_prior  = np.random.default_rng(seed + 9_999)
    historial  : dict = {}
    pids_todos = df_eventos["paciente_id"].unique()
    for pid in pids_todos:
        perfil_p = _perfil_sintetico(pid, seed)
        lam = 0.60
        if perfil_p["tramo_fonasa"] == "A":
            lam += 0.20
        if "RURAL" in perfil_p["zona_geografica"]:
            lam += 0.15
        if not perfil_p["acceso_internet"]:
            lam += 0.10
        prior = int(rng_prior.poisson(lam))
        if prior > 0:
            historial[pid] = prior

    decisiones      : list = []
    rebotes_nsp     : list = []
    n_por_escenario  = {"A_doble": 0, "B_llamada": 0, "C_digital": 0, "ninguna": 0}
    costo_por_esc    = {"A_doble": 0, "B_llamada": 0, "C_digital": 0}

    for idx, ev in df_eventos.iterrows():
        pid    = ev["paciente_id"]
        perfil = _perfil_sintetico(pid, seed)

        via = str(ev.get("via_clinica", via_clinica_default) or via_clinica_default).upper()
        if via not in ("AMARILLA", "ROJA", "VERDE"):
            via = via_clinica_default

        estado = {
            "paciente_id":             pid,
            "historial_inasistencias": historial.get(pid, 0),
            "via_clinica":             via,
            "modalidad":               ev.get("modalidad", "telemedicina"),
            "dias_sin_registro":       int(np.random.default_rng(seed + idx + 7).integers(0, 91)),
            **perfil,
        }

        score   = predictor.calcular_score(estado)
        esc, canal, costo, reduccion = predictor.decidir_intervencion(
            score, estado["zona_geografica"], via_clinica=via
        )

        n_por_escenario[esc] += 1
        if esc != "ninguna":
            costo_por_esc[esc] += costo

        p_base  = p_rebote_via_override.get(via, 0.10)
        p_post  = p_base * (1.0 - reduccion)
        asistio = bool(rng_asistencia.random() > p_post)

        decisiones.append({
            "paciente_id":            pid,
            "control_num":            int(ev.get("control_num", 0)),
            "via_clinica":            via,
            "modalidad":              estado["modalidad"],
            "zona_geografica":        estado["zona_geografica"],
            "tramo_fonasa":           estado["tramo_fonasa"],
            "hba1c":                  estado["hba1c_actual"],
            "edad":                   estado["edad"],
            "historial_prev":         historial.get(pid, 0),
            "score_nsp":              score.score_total,
            "categoria_riesgo":       score.categoria,
            "escenario_aplicado":     esc,
            "canal_notificacion":     canal,
            "costo_intervencion_clp": costo,
            "reduccion_aplicada":     round(reduccion, 4),
            "p_rebote_base":          round(p_base, 4),
            "p_rebote_post":          round(p_post, 4),
            "asistio_final":          asistio,
        })

        if not asistio:
            historial[pid] = historial.get(pid, 0) + 1
            rebotes_nsp.append({
                "paciente_id":        pid,
                "via_clinica_origen": via,
                "categoria_riesgo":   score.categoria,
                "escenario_previo":   esc,
                "costo_clp":          COSTO_REBOTE,
            })

    df_dec = pd.DataFrame(decisiones)
    df_reb = pd.DataFrame(rebotes_nsp)

    dist_cat = df_dec.groupby("categoria_riesgo").agg(
        n_eventos          = ("paciente_id", "count"),
        score_promedio     = ("score_nsp", "mean"),
        tasa_asistencia    = ("asistio_final", "mean"),
        costo_total_interv = ("costo_intervencion_clp", "sum"),
    ).reset_index()

    n_ctrl = len(decisiones)
    costo_intervenciones = sum(costo_por_esc.values())
    costo_rebotes_nsp    = int(len(df_reb) * COSTO_REBOTE)

    df_dec.to_csv(DIR_REPORTS / f"simulacion_{nombre_escenario}_decisiones{sufijo}.csv", index=False)
    df_reb.to_csv(DIR_REPORTS / f"simulacion_{nombre_escenario}_rebotes_nuevos{sufijo}.csv", index=False)
    dist_cat.to_csv(DIR_REPORTS / f"simulacion_{nombre_escenario}_distribucion_categorias{sufijo}.csv", index=False)

    return {
        "n_eventos_control_evaluados": n_ctrl,
        "n_intervenciones_A_doble":    n_por_escenario["A_doble"],
        "n_intervenciones_B_llamada":  n_por_escenario["B_llamada"],
        "n_intervenciones_C_digital":  n_por_escenario["C_digital"],
        "n_sin_intervencion":          n_por_escenario["ninguna"],
        "pct_criticos":  round(100 * n_por_escenario["A_doble"]  / n_ctrl, 1) if n_ctrl else 0,
        "pct_altos":     round(100 * n_por_escenario["B_llamada"] / n_ctrl, 1) if n_ctrl else 0,
        "pct_medios":    round(100 * n_por_escenario["C_digital"] / n_ctrl, 1) if n_ctrl else 0,
        "pct_bajos":     round(100 * n_por_escenario["ninguna"]   / n_ctrl, 1) if n_ctrl else 0,
        "costo_A_doble_clp":        costo_por_esc["A_doble"],
        "costo_B_llamada_clp":      costo_por_esc["B_llamada"],
        "costo_C_digital_clp":      costo_por_esc["C_digital"],
        "costo_intervenciones_clp": int(costo_intervenciones),
        "n_rebotes_nsp":            len(df_reb),
        "costo_rebotes_nsp_clp":    costo_rebotes_nsp,
    }

def simular_capa_nsp_base(seed: int = 42, sufijo: str = "") -> dict:
    archivo_base = DIR_REPORTS / f"simulacion_base_eventos{sufijo}.csv"
    if not archivo_base.exists():
        raise FileNotFoundError(
            f"Falta: {archivo_base}\n"
            "Correr: python src/10_simulador_eventos.py --escenario base "
            "--n-pacientes 500 --dias 365 --seed 42"
        )

    df_eventos = pd.read_csv(archivo_base)

    mask_ctrl    = df_eventos["modalidad"] == "PRESENCIAL_CONTROL"
    mask_prof    = df_eventos["modalidad"] == "PRESENCIAL_PROFESIONAL_NO_DISPONIBLE"
    mask_noshows = df_eventos["modalidad"] == "PRESENCIAL_NO_SHOW"

    eventos_ctrl = df_eventos[mask_ctrl].reset_index(drop=True)

    metricas_nsp = _aplicar_nsp_sobre_eventos(
        df_eventos=eventos_ctrl,
        seed=seed,
        via_clinica_default="AMARILLA",
        p_rebote_via_override=P_REBOTE_VIA,
        nombre_escenario="solo_nsp",
        sufijo=sufijo,
    )

    costo_ctrl         = int(df_eventos[mask_ctrl]["costo_clp"].sum())
    costo_prof_unavail = int(df_eventos[mask_prof]["costo_clp"].sum())
    costo_noshows_base = int(df_eventos[mask_noshows]["costo_clp"].sum())

    mask_fase1  = df_eventos["modalidad"].isin(
        ["PRESENCIAL_INGRESO", "PRESENCIAL_REVISION", "TELEMEDICINA_REVISION"])
    costo_fase1 = int(df_eventos[mask_fase1]["costo_clp"].sum())

    costo_total_nsp = (costo_ctrl
                       + costo_prof_unavail
                       + costo_fase1
                       + metricas_nsp["costo_intervenciones_clp"]
                       + metricas_nsp["costo_rebotes_nsp_clp"])
    n_pac = df_eventos["paciente_id"].nunique()

    metricas = {
        "escenario":                  "solo_nsp",
        "n_pacientes":                n_pac,
        "costo_total_clp":            costo_total_nsp,
        "costo_fase1_clp":            costo_fase1,
        "costo_ctrl_clp":             costo_ctrl,
        "costo_prof_unavail_clp":     costo_prof_unavail,
        "costo_noshows_base_clp":     costo_noshows_base,
        "ahorro_vs_baseline_pct":     round(100 * (BASELINE_CLP - costo_total_nsp) / BASELINE_CLP, 2),
        **metricas_nsp,
    }

    pd.DataFrame([metricas]).to_csv(
        DIR_REPORTS / f"simulacion_solo_nsp_resumen{sufijo}.csv", index=False
    )
    return metricas

def simular_capa_nsp(seed: int = 42, sufijo: str = "") -> dict:
    archivo_base = DIR_REPORTS / f"simulacion_solo_ews_eventos_completos{sufijo}.csv"
    if not archivo_base.exists():
        raise FileNotFoundError(
            f"Falta: {archivo_base}\n"
            "Correr: python src/10_simulador_eventos.py --escenario solo_ews "
            "--n-pacientes 500 --dias 365 --seed 42"
        )

    df_eventos = pd.read_csv(archivo_base)

    tipos_invalidos = ["inasistencia_ews", "reprogramacion_pool_saturado", "rebote"]
    mask_invalido = (
        df_eventos["tipo_evento"].isin(tipos_invalidos)
        if "tipo_evento" in df_eventos.columns
        else pd.Series(False, index=df_eventos.index)
    )

    mask_ctrl = df_eventos["via_clinica"].isin(["AMARILLA", "ROJA"]) & ~mask_invalido
    eventos_ctrl = df_eventos[mask_ctrl].reset_index(drop=True)

    metricas_nsp = _aplicar_nsp_sobre_eventos(
        df_eventos=eventos_ctrl,
        seed=seed,
        via_clinica_default="AMARILLA",
        p_rebote_via_override=P_REBOTE_VIA,
        nombre_escenario="ews_nsp",
        sufijo=sufijo,
    )

    costo_eventos_validos = int(df_eventos[~mask_invalido]["costo_clp"].sum())
    costo_total_nsp = (costo_eventos_validos
                       + metricas_nsp["costo_intervenciones_clp"]
                       + metricas_nsp["costo_rebotes_nsp_clp"])

    ews_csv = DIR_REPORTS / f"simulacion_solo_ews_resumen{sufijo}.csv"
    if ews_csv.exists():
        _r = pd.read_csv(ews_csv).iloc[0]
        costo_ews_op = int(_r.get("costo_eventos_total_clp", _r.get("costo_total_clp", 27_528_000)))
    else:
        costo_ews_op = 27_528_000

    metricas = {
        "escenario":                        "ews_nsp",
        "costo_total_clp":                  costo_total_nsp,
        "costo_eventos_clp":                costo_eventos_validos,
        "tasa_no_asistencia_pct":           round(100 * metricas_nsp["n_rebotes_nsp"]
                                                  / max(metricas_nsp["n_eventos_control_evaluados"], 1), 2),
        "ahorro_vs_baseline_pct":           round(100 * (BASELINE_CLP - costo_total_nsp) / BASELINE_CLP, 2),
        "ahorro_vs_solo_ews_pct":           round(100 * (costo_ews_op - costo_total_nsp) / max(costo_ews_op, 1), 2),
        **metricas_nsp,
    }

    pd.DataFrame([metricas]).to_csv(
        DIR_REPORTS / f"simulacion_ews_nsp_resumen{sufijo}.csv", index=False
    )
    return metricas


def generar_tabla_comparativa_2x2(
    m_base: dict | None = None,
    m_solo_ews: dict | None = None,
    m_solo_nsp: dict | None = None,
    m_ews_nsp:  dict | None = None,
) -> pd.DataFrame:
    def _cargar(archivo: str, fallback) -> dict:
        ruta = DIR_REPORTS / archivo
        if ruta.exists():
            return pd.read_csv(ruta).iloc[0].to_dict()
        return fallback or {}

    entradas = [
        ("base",     "simulacion_base_resumen.csv",      m_base,     "Base (100% presencial)"),
        ("solo_ews", "simulacion_solo_ews_resumen.csv",  m_solo_ews, "Solo EWS (sin NSP)"),
        ("solo_nsp", "simulacion_solo_nsp_resumen.csv",  m_solo_nsp, "Solo NSP (sin EWS)"),
        ("ews_nsp",  "simulacion_ews_nsp_resumen.csv",   m_ews_nsp,  "EWS + NSP"),
    ]

    datos_base = _cargar("simulacion_base_resumen.csv", m_base)
    costo_base = int(datos_base.get("costo_total_clp", datos_base.get("costo_eventos_total_clp", 0)))

    filas = []
    for escenario_id, archivo, metricas_arg, label in entradas:
        datos = _cargar(archivo, metricas_arg)
        costo = int(datos.get("costo_total_clp", datos.get("costo_eventos_total_clp", 0)))
        ahorro = (round(100 * (costo_base - costo) / costo_base, 1)
                  if costo_base > 0 else "")
        filas.append({
            "escenario":              label,
            "escenario_id":           escenario_id,
            "costo_total_clp":        costo,
            "ahorro_vs_base_pct":     ahorro,
        })

    df = pd.DataFrame(filas)
    df.to_csv(DIR_REPORTS / "tabla_comparativa_2x2.csv", index=False)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulador NSP - escenarios solo_nsp y ews_nsp")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-replicas", type=int, default=1,
                        help="Replicas con semillas seed, seed+1, ..., seed+n-1. Default=1.")
    args = parser.parse_args()

    semillas = [args.seed + i for i in range(args.n_replicas)]

    for idx, semilla in enumerate(semillas):
        if args.n_replicas > 1:
            print(f"\n[Replica {idx+1}/{args.n_replicas}] seed={semilla}")
        seed_suf = f"_seed{semilla}" if args.n_replicas > 1 else ""

        print(f"\nDiseño factorial 2x2 - capa NSP (seed={semilla})")

        print("\n[1/2] ews_nsp: lo genera el simulador base")
        m_ews = None

        print("\n[2/2] simulando solo_nsp")
        try:
            m_base_nsp = simular_capa_nsp_base(seed=semilla, sufijo=seed_suf)
            print(f"  Costo total solo_nsp:  ${m_base_nsp['costo_total_clp']:>13,} CLP")
            print(f"  Ahorro vs baseline:     {m_base_nsp['ahorro_vs_baseline_pct']:.1f}%")
        except FileNotFoundError as exc:
            print(f"  [OMITIDO] {exc}")
            m_base_nsp = None

        if args.n_replicas == 1 and m_ews is not None:
            print(f"\nDistribución de categorías solo_nsp (n={m_ews['n_eventos_control_evaluados']} controles)")
            print(f"  CRITICO -> A (doble):   {m_ews['n_intervenciones_A_doble']:>4}  ({m_ews['pct_criticos']:>5.1f}%)")
            print(f"  ALTO    -> B (llamada): {m_ews['n_intervenciones_B_llamada']:>4}  ({m_ews['pct_altos']:>5.1f}%)")
            print(f"  MEDIO   -> C (digital): {m_ews['n_intervenciones_C_digital']:>4}  ({m_ews['pct_medios']:>5.1f}%)")
            print(f"  BAJO    -> sin interv.: {m_ews['n_sin_intervencion']:>4}  ({m_ews['pct_bajos']:>5.1f}%)")
            df_2x2 = generar_tabla_comparativa_2x2(m_solo_nsp=m_base_nsp, m_ews_nsp=m_ews)
            print("\nTabla 2x2")
            print(df_2x2.to_string(index=False))
            print()
