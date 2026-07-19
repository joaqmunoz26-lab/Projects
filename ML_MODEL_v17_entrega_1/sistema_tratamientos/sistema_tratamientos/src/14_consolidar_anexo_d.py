import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

_DIR_BASE    = Path(__file__).resolve().parent.parent
_DIR_REPORTS = _DIR_BASE / "reports"
_DIR_OUT     = _DIR_REPORTS / "evaluacion_final"

VALOR_HORA_CLP              = 2962
MINUTOS_AHORRO_TELECONSULTA = 75
MINUTOS_AHORRO_ASINCRONICA  = 95
FACTOR_EXTRAPOLACION        = 220
COSTO_CONSULTA_TELEMEDICINA = 8630
COSTO_CONSULTA_PRESENCIAL   = 10380
HOSPITALIZACION_MIN_POR_500 = 3_000_000
HOSPITALIZACION_MAX_POR_500 = 7_000_000

TIPOS_CONTROL_EFECTIVO = ("control_via_verde", "control_via_amarilla", "control_via_roja")
TIPO_INASISTENCIA      = "inasistencia_ews"
TIPO_INTERVENCION_NSP  = "intervencion_nsp"
VIAS_VALIDAS           = {"VERDE", "AMARILLA", "ROJA"}
COSTOS_NSP_VALIDOS     = {0, 35, 75, 700, 735, 775}

ORDEN_ESCENARIOS = ("base", "solo_nsp", "solo_ews", "ews_nsp")
ORDEN_VIAS       = ("verde", "amarilla_tele", "amarilla_presencial", "roja")
ETIQUETA_VIA = {
    "verde":               "Via Verde asincronica",
    "amarilla_tele":       "Via Amarilla teleconsulta",
    "amarilla_presencial": "Via Amarilla presencial contingencia",
    "roja":                "Via Roja presencial",
}

GOLDEN_TABLA_17 = {"base": (1658, 178), "solo_nsp": (1594, 114),
                   "solo_ews": (1517, 66), "ews_nsp": (1517, 59)}
GOLDEN_TABLA_20_AHORRO_TOTAL = {"solo_nsp": 696538, "solo_ews": 11054575, "ews_nsp": 11072675}
GOLDEN_TABLA_21_EWS_NSP = {"verde": (651, 651), "amarilla_tele": (397, 363),
                           "amarilla_presencial": (94, 85), "roja": (375, 358)}
GOLDEN_TABLA_22 = {"previstas": 65, "sin_intervencion": 41, "intervenidas": 24,
                   "efectivas": 6, "no_efectivas": 18, "finales": 59}
GOLDEN_TABLA_18 = {"ahorro_directo_500": 11072675, "lucro_cesante_500": 4397089}


def detener_por_integridad(mensaje):
    raise SystemExit(f"[STOP INTEGRIDAD] {mensaje}")


def seeds_del_experimento(seed_inicial, n_replicas):
    return list(range(seed_inicial, seed_inicial + n_replicas))


def ruta_evento_ews(dir_reports, escenario, seed):
    return dir_reports / f"simulacion_{escenario}_eventos_completos_seed{seed}.csv"


def ruta_evento_base(dir_reports, seed):
    return dir_reports / f"simulacion_base_eventos_seed{seed}.csv"


def inasistencias_solo_nsp_desde_resumen(dir_reports, seed):
    resumen = pd.read_csv(dir_reports / f"simulacion_solo_nsp_resumen_seed{seed}.csv").iloc[0]
    return int(resumen["n_rebotes_nsp"])


def validar_integridad_inasistencias_ews(df, escenario, seed):
    inasistencias = df[df["tipo_evento"] == TIPO_INASISTENCIA]
    vias_presentes = set(inasistencias["via_clinica"].astype(str).str.upper().unique())
    if not vias_presentes.issubset(VIAS_VALIDAS):
        detener_por_integridad(
            f"{escenario} seed{seed}: via_clinica invalida en inasistencias -> "
            f"{sorted(vias_presentes - VIAS_VALIDAS)}")
    inasistencias_amarilla = inasistencias[
        inasistencias["via_clinica"].astype(str).str.upper() == "AMARILLA"]
    costos_amarilla = set(inasistencias_amarilla["costo_consulta_clp"].dropna().unique())
    if not costos_amarilla.issubset({COSTO_CONSULTA_TELEMEDICINA, COSTO_CONSULTA_PRESENCIAL}):
        detener_por_integridad(
            f"{escenario} seed{seed}: costo_consulta_clp fuera del decode Amarilla -> "
            f"{sorted(costos_amarilla)}")


def contar_efectivos_e_inasistencias_por_via(df):
    efectivos    = df[df["tipo_evento"].isin(TIPOS_CONTROL_EFECTIVO)]
    inasistencias = df[df["tipo_evento"] == TIPO_INASISTENCIA]
    via_inasistencia = inasistencias["via_clinica"].astype(str).str.upper()
    efectivos_por_via = {
        "verde":               int((efectivos["tipo_evento"] == "control_via_verde").sum()),
        "amarilla_tele":       int(((efectivos["tipo_evento"] == "control_via_amarilla") &
                                     (efectivos["modalidad"] == "telemedicina")).sum()),
        "amarilla_presencial": int(((efectivos["tipo_evento"] == "control_via_amarilla") &
                                     (efectivos["modalidad"] == "presencial_control")).sum()),
        "roja":                int((efectivos["tipo_evento"] == "control_via_roja").sum()),
    }
    inasistencias_por_via = {
        "verde":               int((via_inasistencia == "VERDE").sum()),
        "amarilla_tele":       int(((via_inasistencia == "AMARILLA") &
                                     (inasistencias["costo_consulta_clp"] == COSTO_CONSULTA_TELEMEDICINA)).sum()),
        "amarilla_presencial": int(((via_inasistencia == "AMARILLA") &
                                     (inasistencias["costo_consulta_clp"] == COSTO_CONSULTA_PRESENCIAL)).sum()),
        "roja":                int((via_inasistencia == "ROJA").sum()),
    }
    return efectivos_por_via, inasistencias_por_via


def contar_embudo_nsp_por_costo(df):
    intervenciones = df[df["tipo_evento"] == TIPO_INTERVENCION_NSP]
    costos_presentes = set(int(c) for c in intervenciones["costo_clp"].unique())
    if not costos_presentes.issubset(COSTOS_NSP_VALIDOS):
        detener_por_integridad(
            f"intervencion_nsp con costo fuera de {sorted(COSTOS_NSP_VALIDOS)} -> "
            f"{sorted(costos_presentes)}")
    return {
        "previstas":        int(len(intervenciones)),
        "sin_intervencion": int((intervenciones["costo_clp"] == 0).sum()),
        "intervenidas":     int((intervenciones["costo_clp"] > 0).sum()),
        "finales":          int((df["tipo_evento"] == TIPO_INASISTENCIA).sum()),
    }


def contar_base(df):
    modalidad = df["modalidad"]
    return {
        "efectivo_rutina": int((modalidad == "PRESENCIAL_CONTROL").sum()),
        "inasistencias":   int((modalidad == "PRESENCIAL_NO_SHOW").sum()),
    }


def promedio_de_dicts(lista_dicts):
    claves = lista_dicts[0].keys()
    return {clave: float(np.mean([d[clave] for d in lista_dicts])) for clave in claves}


def agregar_conteos_de_eventos(dir_reports, seeds):
    base_reps, solo_nsp_reps = [], []
    solo_ews_efectivos, solo_ews_inasistencias = [], []
    ews_nsp_efectivos, ews_nsp_inasistencias, ews_nsp_embudo = [], [], []
    for seed in seeds:
        base_df = pd.read_csv(ruta_evento_base(dir_reports, seed))
        base_conteo = contar_base(base_df)
        base_reps.append(base_conteo)
        solo_nsp_reps.append({
            "efectivo_rutina": base_conteo["efectivo_rutina"],
            "inasistencias":   inasistencias_solo_nsp_desde_resumen(dir_reports, seed),
        })
        solo_ews_df = pd.read_csv(ruta_evento_ews(dir_reports, "solo_ews", seed))
        validar_integridad_inasistencias_ews(solo_ews_df, "solo_ews", seed)
        efec_se, inas_se = contar_efectivos_e_inasistencias_por_via(solo_ews_df)
        solo_ews_efectivos.append(efec_se)
        solo_ews_inasistencias.append(inas_se)
        ews_nsp_df = pd.read_csv(ruta_evento_ews(dir_reports, "ews_nsp", seed))
        validar_integridad_inasistencias_ews(ews_nsp_df, "ews_nsp", seed)
        efec_en, inas_en = contar_efectivos_e_inasistencias_por_via(ews_nsp_df)
        ews_nsp_efectivos.append(efec_en)
        ews_nsp_inasistencias.append(inas_en)
        ews_nsp_embudo.append(contar_embudo_nsp_por_costo(ews_nsp_df))
    return {
        "base":               promedio_de_dicts(base_reps),
        "solo_nsp":           promedio_de_dicts(solo_nsp_reps),
        "solo_ews_efec":      promedio_de_dicts(solo_ews_efectivos),
        "solo_ews_inas":      promedio_de_dicts(solo_ews_inasistencias),
        "ews_nsp_efec":       promedio_de_dicts(ews_nsp_efectivos),
        "ews_nsp_inas":       promedio_de_dicts(ews_nsp_inasistencias),
        "ews_nsp_embudo":     promedio_de_dicts(ews_nsp_embudo),
    }


def medias_de_costo_por_componente(dir_reports, seeds):
    modulo_desglose = import_module("16_desglose_costos_escenario")
    modulo_desglose.DIR_REPORTS = dir_reports
    reconstruir_por_escenario = modulo_desglose._RECON
    componentes = modulo_desglose._COMP
    acumulado = {esc: {c: [] for c in componentes} for esc in ORDEN_ESCENARIOS}
    for seed in seeds:
        for esc in ORDEN_ESCENARIOS:
            descomposicion = reconstruir_por_escenario[esc](seed)
            for c in componentes:
                acumulado[esc][c].append(descomposicion[c])
    return {esc: {c: float(np.mean(acumulado[esc][c])) for c in componentes}
            for esc in ORDEN_ESCENARIOS}


def costo_reprogramacion(componentes):
    return (componentes["no_show"] + componentes["prof_no_disponible"]
            + componentes["reprog_pool_saturado"] + componentes["admin"]
            + componentes["intervencion_nsp"] + componentes["inasistencia_post_nsp"])


def tabla_20_ahorro_directo(medias_costo):
    base = medias_costo["base"]
    filas = []
    for esc in ORDEN_ESCENARIOS:
        if esc == "base":
            continue
        escenario = medias_costo[esc]
        ahorro_consulta       = round(base["consulta"] - escenario["consulta"])
        ahorro_traslado       = round(base["traslado"] - escenario["traslado"])
        ahorro_reprogramacion = round(costo_reprogramacion(base) - costo_reprogramacion(escenario))
        ahorro_total          = ahorro_consulta + ahorro_traslado + ahorro_reprogramacion
        filas.append({
            "escenario":              esc,
            "ahorro_consulta":        ahorro_consulta,
            "ahorro_traslado":        ahorro_traslado,
            "ahorro_reprogramacion":  ahorro_reprogramacion,
            "ahorro_total":           ahorro_total,
        })
    return pd.DataFrame(filas)


def celda_prog_efec(programados, efectivos):
    return f"{round(programados)} / {round(efectivos)}"


def total_fase2_ews(efectivos_por_via, inasistencias_por_via):
    programados  = sum(efectivos_por_via[v] + inasistencias_por_via[v] for v in ORDEN_VIAS)
    inasistencias = sum(inasistencias_por_via[v] for v in ORDEN_VIAS)
    efectivos_por_diferencia = programados - inasistencias
    return programados, efectivos_por_diferencia


def tabla_21_prog_efec_por_via(conteos):
    base = conteos["base"]
    nsp  = conteos["solo_nsp"]
    filas = [{
        "metrica":  "Presencial rutina (sin EWS)",
        "base":     celda_prog_efec(base["efectivo_rutina"] + base["inasistencias"], base["efectivo_rutina"]),
        "solo_nsp": celda_prog_efec(nsp["efectivo_rutina"] + nsp["inasistencias"], nsp["efectivo_rutina"]),
        "solo_ews": "-",
        "ews_nsp":  "-",
    }]
    for via in ORDEN_VIAS:
        efec_se = conteos["solo_ews_efec"][via]
        inas_se = conteos["solo_ews_inas"][via]
        efec_en = conteos["ews_nsp_efec"][via]
        inas_en = conteos["ews_nsp_inas"][via]
        filas.append({
            "metrica":  ETIQUETA_VIA[via],
            "base":     "-",
            "solo_nsp": "-",
            "solo_ews": celda_prog_efec(efec_se + inas_se, efec_se),
            "ews_nsp":  celda_prog_efec(efec_en + inas_en, efec_en),
        })
    prog_se, efec_se = total_fase2_ews(conteos["solo_ews_efec"], conteos["solo_ews_inas"])
    prog_en, efec_en = total_fase2_ews(conteos["ews_nsp_efec"], conteos["ews_nsp_inas"])
    filas.append({
        "metrica":  "Total Fase 2",
        "base":     celda_prog_efec(base["efectivo_rutina"] + base["inasistencias"], base["efectivo_rutina"]),
        "solo_nsp": celda_prog_efec(nsp["efectivo_rutina"] + nsp["inasistencias"], nsp["efectivo_rutina"]),
        "solo_ews": celda_prog_efec(prog_se, efec_se),
        "ews_nsp":  celda_prog_efec(prog_en, efec_en),
    })
    filas.append({
        "metrica":  "Inasistencias",
        "base":     round(base["inasistencias"]),
        "solo_nsp": round(nsp["inasistencias"]),
        "solo_ews": round(sum(conteos["solo_ews_inas"].values())),
        "ews_nsp":  round(sum(conteos["ews_nsp_inas"].values())),
    })
    return pd.DataFrame(filas)


def tabla_17_prog_inasist(conteos):
    base = conteos["base"]
    nsp = conteos["solo_nsp"]
    filas = [
        {"escenario": "base",
         "programados":  round(base["efectivo_rutina"] + base["inasistencias"]),
         "inasistencias": round(base["inasistencias"])},
        {"escenario": "solo_nsp",
         "programados":  round(nsp["efectivo_rutina"] + nsp["inasistencias"]),
         "inasistencias": round(nsp["inasistencias"])},
    ]
    for esc, clave_efec, clave_inas in (("solo_ews", "solo_ews_efec", "solo_ews_inas"),
                                        ("ews_nsp",  "ews_nsp_efec",  "ews_nsp_inas")):
        programados  = round(sum(conteos[clave_efec][v] + conteos[clave_inas][v] for v in ORDEN_VIAS))
        inasistencias = round(sum(conteos[clave_inas].values()))
        filas.append({"escenario": esc, "programados": programados, "inasistencias": inasistencias})
    return pd.DataFrame(filas)


def tabla_22_embudo_nsp(embudo_promedio):
    previstas        = round(embudo_promedio["previstas"])
    sin_intervencion = round(embudo_promedio["sin_intervencion"])
    intervenidas     = round(embudo_promedio["intervenidas"])
    finales          = round(embudo_promedio["finales"])
    efectivas        = previstas - finales
    no_efectivas     = intervenidas - efectivas
    return {
        "previstas":        previstas,
        "sin_intervencion": sin_intervencion,
        "intervenidas":     intervenidas,
        "efectivas":        efectivas,
        "no_efectivas":     no_efectivas,
        "finales":          finales,
    }


def verificar_embudo_reproduce_golden(embudo):
    if (embudo["sin_intervencion"] != GOLDEN_TABLA_22["sin_intervencion"]
            or embudo["intervenidas"] != GOLDEN_TABLA_22["intervenidas"]):
        detener_por_integridad(
            "Embudo NSP por costo no reproduce golden 41/24 -> "
            f"sin_intervencion={embudo['sin_intervencion']} (esperado {GOLDEN_TABLA_22['sin_intervencion']}), "
            f"intervenidas={embudo['intervenidas']} (esperado {GOLDEN_TABLA_22['intervenidas']}). "
            "El criterio costo>0 es incorrecto o la data cambio.")


def tabla_18_extrapolacion(tabla_20, conteos):
    ahorro_directo_500 = int(tabla_20.loc[tabla_20["escenario"] == "ews_nsp", "ahorro_total"].iloc[0])
    teleconsultas_efectivas = round(conteos["ews_nsp_efec"]["amarilla_tele"])
    asincronicas_efectivas  = round(conteos["ews_nsp_efec"]["verde"])
    minutos_ahorrados = (teleconsultas_efectivas * MINUTOS_AHORRO_TELECONSULTA
                         + asincronicas_efectivas * MINUTOS_AHORRO_ASINCRONICA)
    lucro_cesante_500 = round(minutos_ahorrados / 60 * VALOR_HORA_CLP)
    filas = [
        {"componente": "Ahorro directo (consultas y viajes)",
         "por_500": ahorro_directo_500,
         "extrapolado_110000": ahorro_directo_500 * FACTOR_EXTRAPOLACION},
        {"componente": "Lucro cesante evitado del paciente",
         "por_500": lucro_cesante_500,
         "extrapolado_110000": lucro_cesante_500 * FACTOR_EXTRAPOLACION},
        {"componente": "Hospitalizaciones evitables (rango referencial)",
         "por_500": f"{HOSPITALIZACION_MIN_POR_500}-{HOSPITALIZACION_MAX_POR_500}",
         "extrapolado_110000": f"{HOSPITALIZACION_MIN_POR_500 * FACTOR_EXTRAPOLACION}-"
                               f"{HOSPITALIZACION_MAX_POR_500 * FACTOR_EXTRAPOLACION}"},
    ]
    return pd.DataFrame(filas)


def reportar_reconciliacion_golden(tabla_17, tabla_20, tabla_21, embudo, tabla_18, conteos):
    print("\n=== RECONCILIACION vs GOLDEN (soft, no aborta) ===")
    for _, fila in tabla_17.iterrows():
        golden_prog, golden_inas = GOLDEN_TABLA_17[fila["escenario"]]
        print(f"  T17 {fila['escenario']:<9} prog {fila['programados']} (golden {golden_prog}, "
              f"d{fila['programados']-golden_prog:+d}) | inas {fila['inasistencias']} "
              f"(golden {golden_inas}, d{fila['inasistencias']-golden_inas:+d})")
    for _, fila in tabla_20.iterrows():
        golden = GOLDEN_TABLA_20_AHORRO_TOTAL[fila["escenario"]]
        print(f"  T20 {fila['escenario']:<9} total {fila['ahorro_total']} (golden {golden}, "
              f"d{fila['ahorro_total']-golden:+d})  [total = suma de columnas por construccion]")
    for via in ORDEN_VIAS:
        efec = round(conteos["ews_nsp_efec"][via])
        prog = round(conteos["ews_nsp_efec"][via] + conteos["ews_nsp_inas"][via])
        golden_prog, golden_efec = GOLDEN_TABLA_21_EWS_NSP[via]
        print(f"  T21 ews_nsp {via:<20} prog {prog} (golden {golden_prog}, d{prog-golden_prog:+d}) | "
              f"efec {efec} (golden {golden_efec}, d{efec-golden_efec:+d})")
    suma_categorias_efec = sum(round(conteos["ews_nsp_efec"][v]) for v in ORDEN_VIAS)
    total_por_diferencia = (sum(round(conteos["ews_nsp_efec"][v] + conteos["ews_nsp_inas"][v]) for v in ORDEN_VIAS)
                            - round(sum(conteos["ews_nsp_inas"].values())))
    print(f"  T21 nota redondeo: categorias efec suman {suma_categorias_efec}; "
          f"total por (prog - inas) = {total_por_diferencia}; "
          f"diferencia +-1 esperada por redondeo de medias de 30 replicas")
    for clave in ("previstas", "sin_intervencion", "intervenidas", "efectivas", "no_efectivas", "finales"):
        print(f"  T22 {clave:<17} {embudo[clave]} (golden {GOLDEN_TABLA_22[clave]}, "
              f"d{embudo[clave]-GOLDEN_TABLA_22[clave]:+d})")
    directo = int(tabla_18.loc[0, "por_500"])
    lucro   = int(tabla_18.loc[1, "por_500"])
    print(f"  T18 ahorro_directo_500 {directo} (golden {GOLDEN_TABLA_18['ahorro_directo_500']}, "
          f"d{directo-GOLDEN_TABLA_18['ahorro_directo_500']:+d})")
    print(f"  T18 lucro_cesante_500  {lucro} (golden {GOLDEN_TABLA_18['lucro_cesante_500']}, "
          f"d{lucro-GOLDEN_TABLA_18['lucro_cesante_500']:+d})")


def consolidar_anexo(seed_inicial, n_replicas, dir_reports, dir_out):
    dir_out.mkdir(parents=True, exist_ok=True)
    seeds = seeds_del_experimento(seed_inicial, n_replicas)
    conteos = agregar_conteos_de_eventos(dir_reports, seeds)
    medias_costo = medias_de_costo_por_componente(dir_reports, seeds)

    tabla_21 = tabla_21_prog_efec_por_via(conteos)
    tabla_17 = tabla_17_prog_inasist(conteos)
    tabla_20 = tabla_20_ahorro_directo(medias_costo)
    embudo   = tabla_22_embudo_nsp(conteos["ews_nsp_embudo"])
    verificar_embudo_reproduce_golden(embudo)
    tabla_22 = pd.DataFrame([embudo])
    tabla_18 = tabla_18_extrapolacion(tabla_20, conteos)

    tabla_17.to_csv(dir_out / "tabla_17_prog_inasist_por_escenario.csv", index=False)
    tabla_18.to_csv(dir_out / "tabla_18_extrapolacion_regional.csv", index=False)
    tabla_20.to_csv(dir_out / "tabla_20_ahorro_directo.csv", index=False)
    tabla_21.to_csv(dir_out / "tabla_21_prog_efec_por_via.csv", index=False)
    tabla_22.to_csv(dir_out / "tabla_22_embudo_nsp.csv", index=False)

    return {"tabla_17": tabla_17, "tabla_18": tabla_18, "tabla_20": tabla_20,
            "tabla_21": tabla_21, "embudo": embudo, "conteos": conteos}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-replicas",  type=int, default=30)
    parser.add_argument("--dir-reports", type=str, default=str(_DIR_REPORTS))
    parser.add_argument("--dir-out",     type=str, default=str(_DIR_OUT))
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} | pandas {pd.__version__}")
    resultado = consolidar_anexo(args.seed, args.n_replicas,
                                 Path(args.dir_reports), Path(args.dir_out))
    print("\n=== Tabla 17 ===")
    print(resultado["tabla_17"].to_string(index=False))
    print("\n=== Tabla 20 ===")
    print(resultado["tabla_20"].to_string(index=False))
    print("\n=== Tabla 21 ===")
    print(resultado["tabla_21"].to_string(index=False))
    print("\n=== Tabla 22 ===")
    print(pd.DataFrame([resultado["embudo"]]).to_string(index=False))
    print("\n=== Tabla 18 ===")
    print(resultado["tabla_18"].to_string(index=False))
    reportar_reconciliacion_golden(resultado["tabla_17"], resultado["tabla_20"],
                                   resultado["tabla_21"], resultado["embudo"],
                                   resultado["tabla_18"], resultado["conteos"])
