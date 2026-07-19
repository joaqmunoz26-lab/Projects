import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_BASE    = Path(__file__).resolve().parent.parent
DIR_REPORTS = DIR_BASE / "reports"
DIR_FINAL   = DIR_REPORTS / "evaluacion_final"

COLORES = {
    "Base (100% presencial)": "#888780",
    "Solo EWS (sin NSP)":     "#6B2FA0",
    "Solo NSP (sin EWS)":     "#378ADD",
    "EWS + NSP":              "#2E8B57",
}

COSTO_PRESENCIAL    = 28_000
COSTO_TELEMEDICINA  = 12_000
COSTO_ASINCRONA     = 6_000
TIEMPO_PRESENCIAL   = 95
TIEMPO_TELEMEDICINA = 20
TIEMPO_ASINCRONA    = 0

AMARILLA_TELE = 0.81
AMARILLA_PRES = 0.19

_ESCENARIOS_2X2 = [
    ("base",     "Base (100% presencial)"),
    ("solo_ews", "Solo EWS (sin NSP)"),
    ("solo_nsp", "Solo NSP (sin EWS)"),
    ("ews_nsp",  "EWS + NSP"),
]

_EWS_NOMBRES = {"Solo EWS (sin NSP)"}

_NSP_NOMBRES = {"Solo NSP (sin EWS)", "EWS + NSP"}

def cargar_resultados_simulaciones(dir_reports: Path = DIR_REPORTS) -> dict:
    resultados = {}
    for nombre_archivo, nombre_legible in _ESCENARIOS_2X2:
        ruta_resumen   = dir_reports / f"simulacion_{nombre_archivo}_resumen.csv"
        ruta_pacientes = dir_reports / f"simulacion_{nombre_archivo}_pacientes.csv"

        if not ruta_resumen.exists():
            print(f"  [advertencia] {nombre_legible}: no se encontro "
                  f"{ruta_resumen.name}, se omite.")
            continue

        try:
            resumen  = pd.read_csv(ruta_resumen)
            pacientes = (pd.read_csv(ruta_pacientes)
                         if ruta_pacientes.exists() else pd.DataFrame())
            resultados[nombre_legible] = {
                "resumen":        resumen.iloc[0].to_dict() if len(resumen) > 0 else {},
                "pacientes":      pacientes,
                "archivo_origen": nombre_archivo,
            }
            print(f"  [ok] {nombre_legible}")
        except Exception as e:
            print(f"  [error] {nombre_legible}: {e}")

    return resultados

def estimar_costo_ews(resumen_ews: dict) -> float:
    via_verde    = resumen_ews.get("fase2_via_verde",    0)
    via_amarilla = resumen_ews.get("fase2_via_amarilla", 0)
    via_roja     = resumen_ews.get("fase2_via_roja",     0)

    costo = (via_verde    * COSTO_ASINCRONA
             + via_amarilla * (AMARILLA_TELE * COSTO_TELEMEDICINA
                               + AMARILLA_PRES * COSTO_PRESENCIAL)
             + via_roja    * COSTO_PRESENCIAL)
    return round(costo, 0)

def estimar_tiempo_ews(resumen_ews: dict) -> float:
    via_verde    = resumen_ews.get("fase2_via_verde",    0)
    via_amarilla = resumen_ews.get("fase2_via_amarilla", 0)
    via_roja     = resumen_ews.get("fase2_via_roja",     0)

    tiempo = (via_verde    * TIEMPO_ASINCRONA
              + via_amarilla * (AMARILLA_TELE * TIEMPO_TELEMEDICINA
                                + AMARILLA_PRES * TIEMPO_PRESENCIAL)
              + via_roja    * TIEMPO_PRESENCIAL)
    return round(tiempo, 0)

def construir_tabla_comparativa(resultados: dict) -> pd.DataFrame:
    filas = []

    for nombre, datos in resultados.items():
        resumen = datos["resumen"]

        if nombre in _EWS_NOMBRES:
            n_pac       = resumen.get("total_procesados", 0)
            confirmados = resumen.get("fase1_confirmados", 0)

            costo_ev = resumen.get("costo_eventos_total_clp")
            tiempo_ev = resumen.get("tiempo_paciente_total_min")
            if costo_ev is not None:
                costo  = int(costo_ev)
                tiempo = int(tiempo_ev) if tiempo_ev is not None else estimar_tiempo_ews(resumen)
                modelo_costos = "eventos_completos"
            else:
                costo  = estimar_costo_ews(resumen)
                tiempo = estimar_tiempo_ews(resumen)
                modelo_costos = "decisiones_individuales"

            fila = {
                "escenario":               nombre,
                "n_pacientes_procesados":  n_pac,
                "dm2_confirmados":         confirmados,
                "tasa_confirmacion_pct":   round(100 * confirmados / n_pac, 1) if n_pac else 0,
                "costo_total_clp":         costo,
                "tiempo_total_min":        tiempo,
                "costo_por_confirmado":    round(costo / confirmados) if confirmados else 0,
                "tiempo_por_confirmado":   round(tiempo / confirmados) if confirmados else 0,
                "modelo_costos":           modelo_costos,
                "via_verde":               resumen.get("fase2_via_verde",    0),
                "via_amarilla":            resumen.get("fase2_via_amarilla", 0),
                "via_roja":                resumen.get("fase2_via_roja",     0),
                "bypass_barrera_duras":    resumen.get("barrera_duras_bypass", 0),
                "uso_fallback_red":        resumen.get("uso_fallback_red",   0),
                "hitl_encolado":           resumen.get("hitl_encolado",      0),
                "hitl_escalado_t48":       resumen.get("hitl_escalado_t48",  0),
                "hitl_escalado_t72":       resumen.get("hitl_escalado_t72",  0),
                "hitl_escalado_t96":       resumen.get("hitl_escalado_t96_emergencia", 0),
                "control_adelantado":      resumen.get("control_rutina_adelantado", 0),
                "hitl_aprobadas_humano":   resumen.get("aprobadas_por_humano",    0),
                "hitl_rechazadas_humano":  resumen.get("rechazadas_por_humano",   0),
                "eventos_completos_total": resumen.get("eventos_completos_total", 0),
                "rebotes_generados":       resumen.get("rebotes_generados",       0),
                "tasa_rebote_pct":         resumen.get("tasa_rebote_pct",         0),
                "transiciones_totales":    resumen.get("transiciones_totales", 0),
                "pacientes_con_via_verde": resumen.get("pacientes_con_via_verde_alguna_vez", 0),
                "mejoraron_de_via":        resumen.get("pacientes_que_mejoraron_de_via", 0),
                "empeoraron_de_via":       resumen.get("pacientes_que_empeoraron_de_via", 0),
                "presenciales": (resumen.get("fase2_via_roja", 0)
                                 + resumen.get("fase2_via_amarilla", 0) * AMARILLA_PRES),
                "telemedicina": resumen.get("fase2_via_amarilla", 0) * AMARILLA_TELE,
                "sin_consulta": resumen.get("fase2_via_verde", 0),
                "rebotes":      resumen.get("rebotes_generados", 0),
            }

        elif nombre in _NSP_NOMBRES:
            n_pac = resumen.get("n_pacientes", 0)
            costo = int(resumen.get("costo_total_clp", 0))
            fila = {
                "escenario":              nombre,
                "n_pacientes_procesados": n_pac,
                "dm2_confirmados":        n_pac,
                "tasa_confirmacion_pct":  100.0,
                "costo_total_clp":        costo,
                "tiempo_total_min":       0,
                "costo_por_confirmado":   round(costo / n_pac) if n_pac else 0,
                "tiempo_por_confirmado":  0,
                "modelo_costos":          "nsp_post_procesamiento",
                "n_intervenciones_A":     resumen.get("n_intervenciones_A_doble",  0),
                "n_intervenciones_B":     resumen.get("n_intervenciones_B_llamada", 0),
                "n_intervenciones_C":     resumen.get("n_intervenciones_C_digital", 0),
                "n_rebotes_nsp":          resumen.get("n_rebotes_nsp", 0),
                "presenciales":           0,
                "telemedicina":           0,
                "sin_consulta":           0,
                "rebotes":                resumen.get("n_rebotes_nsp", 0),
            }

        else:
            n_pac  = resumen.get("n_pacientes", 0)
            costo  = resumen.get("costo_total_clp",  0)
            tiempo = resumen.get("tiempo_total_min", 0)
            fila = {
                "escenario":              nombre,
                "n_pacientes_procesados": n_pac,
                "dm2_confirmados":        n_pac,
                "tasa_confirmacion_pct":  100.0,
                "costo_total_clp":        costo,
                "tiempo_total_min":       tiempo,
                "costo_por_confirmado":   round(costo / n_pac) if n_pac else 0,
                "tiempo_por_confirmado":  round(tiempo / n_pac) if n_pac else 0,
                "modelo_costos":          "eventos_logisticos",
                "presenciales": resumen.get("total_presenciales", 0),
                "telemedicina": resumen.get("total_telemedicina", 0),
                "sin_consulta": resumen.get("total_sin_consulta", 0),
                "rebotes":      resumen.get("rebotes_totales",    0),
            }

        filas.append(fila)

    df = pd.DataFrame(filas)

    mask_base = df["escenario"] == "Base (100% presencial)"
    if mask_base.any():
        base_costo  = df.loc[mask_base, "costo_total_clp"].iloc[0]
        base_tiempo = df.loc[mask_base, "tiempo_total_min"].iloc[0]

        df["ahorro_costo_pct_vs_base"] = df["costo_total_clp"].apply(
            lambda c: round(100 * (base_costo - c) / base_costo, 1)
            if base_costo > 0 else None
        )
        df["ahorro_tiempo_pct_vs_base"] = df["tiempo_total_min"].apply(
            lambda t: round(100 * (base_tiempo - t) / base_tiempo, 1)
            if base_tiempo > 0 else None
        )

    return df

def graficar_costos(df: pd.DataFrame, ruta: Path):
    df_s = df.sort_values("costo_total_clp", ascending=True)
    plt.figure(figsize=(10, 5))
    colores = [COLORES.get(e, "gray") for e in df_s["escenario"]]
    barras  = plt.barh(df_s["escenario"],
                       df_s["costo_total_clp"] / 1_000_000,
                       color=colores, edgecolor="white", linewidth=1)
    for b, v in zip(barras, df_s["costo_total_clp"] / 1_000_000):
        plt.text(v + 0.3, b.get_y() + b.get_height() / 2,
                 f"${v:.1f}M", va="center", fontsize=10)
    plt.xlabel("Costo total (millones CLP)", fontsize=11)
    plt.title("Comparativa economica -- diseno factorial 2x2", fontsize=13)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def graficar_tiempos(df: pd.DataFrame, ruta: Path):
    df_plot = df[df["tiempo_total_min"] > 0].sort_values("tiempo_total_min", ascending=True)
    if df_plot.empty:
        return
    plt.figure(figsize=(10, 5))
    colores = [COLORES.get(e, "gray") for e in df_plot["escenario"]]
    barras  = plt.barh(df_plot["escenario"],
                       df_plot["tiempo_total_min"] / 60,
                       color=colores, edgecolor="white", linewidth=1)
    for b, v in zip(barras, df_plot["tiempo_total_min"] / 60):
        plt.text(v + 5, b.get_y() + b.get_height() / 2,
                 f"{v:.0f}h", va="center", fontsize=10)
    plt.xlabel("Tiempo total pacientes (horas)", fontsize=11)
    plt.title("Tiempo invertido por pacientes -- comparativa 2x2", fontsize=13)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def graficar_distribucion_vias(df: pd.DataFrame, ruta: Path):
    plt.figure(figsize=(11, 6))
    nombres    = list(df["escenario"])
    presenciales = list(df["presenciales"].fillna(0))
    telemedicina = list(df["telemedicina"].fillna(0))
    sin_consulta = list(df["sin_consulta"].fillna(0))

    plt.bar(nombres, presenciales, label="Presencial",             color="#D85A30")
    plt.bar(nombres, telemedicina, bottom=presenciales,
            label="Telemedicina",             color="#378ADD")
    bottom_sin = [p + t for p, t in zip(presenciales, telemedicina)]
    plt.bar(nombres, sin_consulta, bottom=bottom_sin,
            label="Sin consulta / Asincrona", color="#97C459")

    plt.ylabel("Eventos / Pacientes", fontsize=11)
    plt.title("Distribucion de modalidades -- diseno 2x2", fontsize=13)
    plt.legend(loc="upper right", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()

def analizar_sensibilidad_parametros(df_principal: pd.DataFrame) -> pd.DataFrame:
    if ("Solo EWS (sin NSP)" not in df_principal["escenario"].values
            or "Base (100% presencial)" not in df_principal["escenario"].values):
        return pd.DataFrame()

    ews_row  = df_principal[df_principal["escenario"] == "Solo EWS (sin NSP)"].iloc[0]
    base_row = df_principal[df_principal["escenario"] == "Base (100% presencial)"].iloc[0]

    costo_ews_base  = ews_row["costo_total_clp"]
    costo_base_base = base_row["costo_total_clp"]
    ahorro_base_pct = (round(100 * (costo_base_base - costo_ews_base) / costo_base_base, 2)
                       if costo_base_base > 0 else 0)

    variaciones = [
        ("tasa_rebote",      [0.10, 0.18, 0.25], 0.18),
        ("costo_presencial", [20_000, 28_000, 40_000], 28_000),
        ("pct_rural",        [0.15, 0.35, 0.50], 0.35),
    ]

    casos = []
    for param, valores, valor_base in variaciones:
        for valor in valores:
            if param == "tasa_rebote":
                factor = 1 + (valor - valor_base) * 0.5
            elif param == "costo_presencial":
                factor = valor / valor_base
            else:
                factor = 1 + (valor - valor_base) * 0.3

            costo_base_aj = costo_base_base * factor
            costo_ews_aj  = costo_ews_base  * (factor * 0.5 + 0.5)

            ahorro_aj = (round(100 * (costo_base_aj - costo_ews_aj) / costo_base_aj, 2)
                         if costo_base_aj > 0 else 0)

            casos.append({
                "parametro_variado": param,
                "valor_probado":     valor,
                "valor_base":        valor_base,
                "ahorro_pct":        ahorro_aj,
                "ahorro_base_pct":   ahorro_base_pct,
            })

    return pd.DataFrame(casos)

def generar_tabla_markdown(df: pd.DataFrame) -> str:
    n_esc = len(df)
    md  = f"# Tabla comparativa final -- {n_esc} escenarios (diseno 2x2)\n\n"
    md += "## Metricas principales\n\n"
    md += ("| Escenario | Costo total (CLP) | Costo/pac | "
           "Tiempo total (min) | Ahorro costo % | Ahorro tiempo % |\n")
    md += "|---|---|---|---|---|---|\n"

    for _, f in df.iterrows():
        costo   = f.get("costo_total_clp", 0)
        costo_c = f.get("costo_por_confirmado", 0)
        tiempo  = f.get("tiempo_total_min", 0)
        ac      = f.get("ahorro_costo_pct_vs_base",  "---")
        at      = f.get("ahorro_tiempo_pct_vs_base", "---")
        ac_str  = f"{ac}%" if ac not in ("---", None) else "---"
        at_str  = f"{at}%" if at not in ("---", None) else "---"
        md += (f"| {f['escenario']} | ${costo:,.0f} | ${costo_c:,.0f} | "
               f"{tiempo:,.0f} | {ac_str} | {at_str} |\n")

    ews_disponibles = [n for n in _EWS_NOMBRES if n in df["escenario"].values]
    if ews_disponibles:
        md += "\n## Metricas EWS especificas\n\n"
        for nombre_ews in ews_disponibles:
            e = df[df["escenario"] == nombre_ews].iloc[0]
            md += f"### {nombre_ews}\n\n"
            md += f"- **Pacientes procesados**: {int(e.get('n_pacientes_procesados', 0))}\n"
            md += f"- **Via Verde**: {int(e.get('via_verde', 0))}\n"
            md += f"- **Via Amarilla**: {int(e.get('via_amarilla', 0))}\n"
            md += f"- **Via Roja**: {int(e.get('via_roja', 0))}\n"
            md += f"- **HITL encolados**: {int(e.get('hitl_encolado', 0))}\n"
            md += f"- **Escalamientos T+96h (Fase C)**: {int(e.get('hitl_escalado_t96', 0))}\n"
            ev_total = int(e.get("eventos_completos_total", 0))
            if ev_total > 0:
                rebotes  = int(e.get("rebotes_generados", 0))
                tasa_rb  = e.get("tasa_rebote_pct", 0)
                md += f"- **Eventos completos (horizonte anual)**: {ev_total}\n"
                md += f"- **Rebotes generados**: {rebotes} ({tasa_rb:.1f}%)\n"
            trans = int(e.get("transiciones_totales", 0))
            if trans > 0:
                verde   = int(e.get("pacientes_con_via_verde", 0))
                mejoran = int(e.get("mejoraron_de_via", 0))
                empeo   = int(e.get("empeoraron_de_via", 0))
                md += f"- **Transiciones de via detectadas**: {trans}\n"
                md += f"- **Pacientes con Via Verde en algun control**: {verde}\n"
                md += f"- **Mejoraron / Empeoraron de via**: {mejoran} / {empeo}\n"
            md += "\n"

    return md

def generar_reporte_completo_tesis(df: pd.DataFrame,
                                   df_sensibilidad: pd.DataFrame,
                                   ruta: Path) -> None:
    solo_ews_disponible = "Solo EWS (sin NSP)" in df["escenario"].values

    md  = "# Capitulo de Resultados -- Evaluacion Economica (diseno factorial 2x2)\n\n"

    n_esc = len(df)
    md += f"## 1. Comparativa de {n_esc} escenarios (diseno 2x2)\n\n"
    tabla_md = generar_tabla_markdown(df)
    tabla_md = tabla_md.split("\n\n", 1)[-1] if "\n\n" in tabla_md else tabla_md
    md += tabla_md
    md += "\n"

    md += "## 2. Analisis de sensibilidad\n\n"
    if not df_sensibilidad.empty:
        md += "| Parametro | Valor | Ahorro vs Base (%) |\n"
        md += "|---|---|---|\n"
        for _, f in df_sensibilidad.iterrows():
            md += f"| {f['parametro_variado']} | {f['valor_probado']} | {f['ahorro_pct']}% |\n"
    else:
        md += "_No disponible (requiere escenarios base y solo_ews)._\n"
    md += "\n"

    md += "## 3. Modelo longitudinal de re-evaluacion trimestral\n\n"
    if solo_ews_disponible:
        e = df[df["escenario"] == "Solo EWS (sin NSP)"].iloc[0]
        trans  = int(e.get("transiciones_totales", 0))
        verde  = int(e.get("pacientes_con_via_verde", 0))
        conf   = int(e.get("dm2_confirmados", 1)) or 1
        mejora = int(e.get("mejoraron_de_via", 0))
        empeo  = int(e.get("empeoraron_de_via", 0))
        costo  = int(e.get("costo_total_clp", 0))
        md += (f"- Transiciones detectadas entre controles: **{trans}**\n"
               f"- Pacientes con Via Verde en algun control: **{verde}** ({100*verde/conf:.1f}%)\n"
               f"- Mejoraron de via (intensidad |): **{mejora}**\n"
               f"- Empeoraron de via (intensidad ): **{empeo}**\n"
               f"- Costo total anual: **${costo:,.0f} CLP**\n\n")

    ruta.write_text(md, encoding="utf-8")

import math as _math


def agregar_replicas(
    escenario_id: str,
    dir_reports: Path,
    n_replicas: int,
    seed: int = 42,
) -> pd.DataFrame:
    archivos = [
        dir_reports / f"simulacion_{escenario_id}_resumen_seed{seed + i}.csv"
        for i in range(n_replicas)
    ]
    faltantes = [str(p) for p in archivos if not p.exists()]
    if faltantes:
        raise ValueError(
            f"Faltan {len(faltantes)} archivo(s) de replicas para '{escenario_id}':\n"
            + "\n".join(f"  {f}" for f in faltantes)
        )

    filas_dfs = [pd.read_csv(p).iloc[0] for p in archivos]
    df_raw    = pd.DataFrame(filas_dfs)

    numericas = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    filas_agg = []
    for col in numericas:
        vals   = df_raw[col].dropna()
        n      = len(vals)
        media  = float(vals.mean())
        sd     = float(vals.std(ddof=1)) if n > 1 else 0.0
        margen = 1.96 * sd / _math.sqrt(n) if n > 0 else 0.0
        filas_agg.append({
            "metrica":    col,
            "media":      round(media, 4),
            "sd":         round(sd, 4),
            "ic95_inf":   round(media - margen, 4),
            "ic95_sup":   round(media + margen, 4),
            "n_replicas": n,
        })

    return pd.DataFrame(filas_agg)

def ejecutar_evaluacion_replicas(
    dir_reports: Path = DIR_REPORTS,
    dir_final: Path   = DIR_FINAL,
    n_replicas: int   = 1,
    seed: int         = 42,
    escenarios: list  = None,
) -> dict:
    if escenarios is None:
        escenarios = [e for e, _ in _ESCENARIOS_2X2]

    dir_final.mkdir(parents=True, exist_ok=True)
    resultados = {}

    print("=" * 60)
    print(f"Agregacion de {n_replicas} replica(s) por escenario (seed base={seed})")
    print("=" * 60)

    for esc_id in escenarios:
        print(f"\n  {esc_id} (seeds {seed}..{seed + n_replicas - 1}) ...", end=" ")
        try:
            df_agg = agregar_replicas(esc_id, dir_reports, n_replicas, seed)
            ruta   = dir_final / f"evaluacion_{esc_id}_agregado.csv"
            df_agg.to_csv(ruta, index=False)
            resultados[esc_id] = df_agg
            media_costo = df_agg.loc[
                df_agg["metrica"] == "costo_total_clp", "media"
            ].values
            if len(media_costo):
                print(f"ok  (costo media: ${media_costo[0]:,.0f} CLP)")
            else:
                print("ok")
        except ValueError as exc:
            print(f"[OMITIDO] {exc}")

    return resultados

def ejecutar_evaluacion_economica(dir_reports: Path = DIR_REPORTS,
                                  dir_final: Path = DIR_FINAL) -> pd.DataFrame:
    print("=" * 60)
    print("Evaluacion Economica Final -- Diseno Factorial 2x2")
    print("=" * 60)

    dir_final.mkdir(parents=True, exist_ok=True)

    print("\n1. Cargando resultados de las simulaciones...")
    resultados = cargar_resultados_simulaciones(dir_reports)

    if len(resultados) < 2:
        print("\nError: se necesitan al menos 2 escenarios.")
        return pd.DataFrame()

    print("\n2. Construyendo tabla comparativa...")
    df = construir_tabla_comparativa(resultados)
    df.to_csv(dir_final / "tabla_comparativa_2x2.csv", index=False)

    print("\n3. Generando graficos...")
    graficar_costos(df,            dir_final / "grafico_costos.png")
    graficar_tiempos(df,           dir_final / "grafico_tiempos.png")
    graficar_distribucion_vias(df, dir_final / "grafico_distribucion_vias.png")

    print("\n4. Analisis de sensibilidad...")
    df_sens = analizar_sensibilidad_parametros(df)
    if not df_sens.empty:
        df_sens.to_csv(dir_final / "sensibilidad_parametros.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Resumen economico final -- {len(df)} escenarios")
    print("=" * 60)
    for _, f in df.sort_values("costo_total_clp").iterrows():
        nombre = f["escenario"]
        costo  = f.get("costo_total_clp", 0)
        ahorro = f.get("ahorro_costo_pct_vs_base", "---")
        print(f"  {nombre:25s}: ${costo:>15,.0f} CLP   ahorro: {ahorro}%")

    print(f"\nArchivos generados en: {dir_final}")
    archivos = [
        "tabla_comparativa_2x2.csv",
        "grafico_costos.png",
        "grafico_tiempos.png",
        "grafico_distribucion_vias.png",
        "sensibilidad_parametros.csv",
    ]
    for a in archivos:
        existe = "[ok]" if (dir_final / a).exists() else "[--]"
        print(f"  {existe} {a}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluador economico final del sistema EWS -- diseno 2x2"
    )
    parser.add_argument("--n-replicas", type=int, default=1,
                        help="Replicas a procesar (1=sin sufijo, N>1=con sufijo _seedX).")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Semilla inicial. Replicas usan seed, seed+1, ...")
    parser.add_argument("--escenarios", nargs="+",
                        default=["base", "solo_ews", "solo_nsp", "ews_nsp"],
                        choices=["base", "solo_ews", "solo_nsp", "ews_nsp"],
                        help="Escenarios a procesar (default: los 4 del 2x2).")
    args = parser.parse_args()

    if args.n_replicas == 1:
        ejecutar_evaluacion_economica()
    else:
        ejecutar_evaluacion_replicas(
            n_replicas=args.n_replicas,
            seed=args.seed,
            escenarios=args.escenarios,
        )
