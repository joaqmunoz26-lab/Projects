import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIR_BASE    = Path(__file__).resolve().parent.parent
DIR_REPORTS = DIR_BASE / "reports"
DIR_OUT     = DIR_REPORTS / "evaluacion_final"

COSTO_CONSULTA_CONTROL_CLP = 10_380
COSTO_ADMIN_REPROG_CLP     = 500
TIPOS_INVALIDOS_NSP        = {"inasistencia_ews", "reprogramacion_pool_saturado", "rebote"}
TIPOS_CONTROL              = {"control_via_amarilla", "control_via_roja", "control_via_verde"}
FASE1_MODALIDADES          = ["PRESENCIAL_INGRESO", "PRESENCIAL_REVISION", "TELEMEDICINA_REVISION"]

_COMP = ("consulta", "traslado", "no_show", "prof_no_disponible",
         "reprog_pool_saturado", "admin", "intervencion_nsp", "inasistencia_post_nsp")


def _resumen_total(esc, s):
    r = pd.read_csv(DIR_REPORTS / f"simulacion_{esc}_resumen_seed{s}.csv").iloc[0]
    for col in ("costo_total_clp", "costo_eventos_total_clp"):
        if col in r.index and pd.notna(r[col]):
            return int(r[col])
    raise KeyError(f"sin columna de costo total en resumen de {esc} seed {s}")


def _interv_inasist(esc, s):
    r = pd.read_csv(DIR_REPORTS / f"simulacion_{esc}_resumen_seed{s}.csv").iloc[0]
    return int(r.get("costo_intervenciones_clp", 0)), int(r.get("costo_rebotes_nsp_clp", 0))


def _cero():
    return {c: 0 for c in _COMP}


def _seed_base(s):
    be   = pd.read_csv(DIR_REPORTS / f"simulacion_base_eventos_seed{s}.csv")
    ctrl = be[be["modalidad"] == "PRESENCIAL_CONTROL"]
    nos  = be[be["modalidad"] == "PRESENCIAL_NO_SHOW"]
    prof = be[be["modalidad"] == "PRESENCIAL_PROFESIONAL_NO_DISPONIBLE"]
    f1   = int(be[be["modalidad"].isin(FASE1_MODALIDADES)]["costo_clp"].sum())
    consulta_ctrl = len(ctrl) * COSTO_CONSULTA_CONTROL_CLP
    d = _cero()
    d["consulta"]           = consulta_ctrl + f1
    d["traslado"]           = int(ctrl["costo_clp"].sum()) - consulta_ctrl
    d["no_show"]            = int(nos["costo_clp"].sum())  - len(nos) * COSTO_ADMIN_REPROG_CLP
    d["prof_no_disponible"] = int(prof["costo_clp"].sum()) - len(prof) * COSTO_ADMIN_REPROG_CLP
    d["admin"]              = (len(nos) + len(prof)) * COSTO_ADMIN_REPROG_CLP
    return d


def _seed_solo_ews(s):
    ec   = pd.read_csv(DIR_REPORTS / f"simulacion_solo_ews_eventos_completos_seed{s}.csv")
    ctl  = ec[ec["tipo_evento"].isin(TIPOS_CONTROL)]
    diag = int(ec[ec["tipo_evento"].isin(["diagnostico_fase1_ingreso", "diagnostico_fase1_revision"])]["costo_clp"].sum())
    ina  = ec[ec["tipo_evento"] == "inasistencia_ews"]
    rep  = ec[ec["tipo_evento"] == "reprogramacion_pool_saturado"]
    d = _cero()
    d["consulta"]             = int(ctl["costo_consulta_clp"].sum()) + diag
    d["traslado"]             = int(ctl["costo_viaje_clp"].sum())
    d["no_show"]              = int(ina["costo_clp"].sum()) - len(ina) * COSTO_ADMIN_REPROG_CLP
    d["reprog_pool_saturado"] = int(rep["costo_clp"].sum()) - len(rep) * COSTO_ADMIN_REPROG_CLP
    d["admin"]                = (len(ina) + len(rep)) * COSTO_ADMIN_REPROG_CLP
    return d


def _seed_solo_nsp(s):
    be   = pd.read_csv(DIR_REPORTS / f"simulacion_base_eventos_seed{s}.csv")
    ctrl = be[be["modalidad"] == "PRESENCIAL_CONTROL"]
    prof = be[be["modalidad"] == "PRESENCIAL_PROFESIONAL_NO_DISPONIBLE"]
    interv, inasist = _interv_inasist("solo_nsp", s)
    f1   = int(be[be["modalidad"].isin(FASE1_MODALIDADES)]["costo_clp"].sum())
    consulta_ctrl = len(ctrl) * COSTO_CONSULTA_CONTROL_CLP
    d = _cero()
    d["consulta"]              = consulta_ctrl + f1
    d["traslado"]              = int(ctrl["costo_clp"].sum()) - consulta_ctrl
    d["prof_no_disponible"]    = int(prof["costo_clp"].sum()) - len(prof) * COSTO_ADMIN_REPROG_CLP
    d["admin"]                 = len(prof) * COSTO_ADMIN_REPROG_CLP
    d["intervencion_nsp"]      = interv
    d["inasistencia_post_nsp"] = inasist
    return d


def _seed_ews_nsp(s):
    ec   = pd.read_csv(DIR_REPORTS / f"simulacion_ews_nsp_eventos_completos_seed{s}.csv")
    ctl  = ec[ec["tipo_evento"].isin(TIPOS_CONTROL)]
    diag = int(ec[ec["tipo_evento"].isin(["diagnostico_fase1_ingreso", "diagnostico_fase1_revision"])]["costo_clp"].sum())
    ina  = ec[ec["tipo_evento"] == "inasistencia_ews"]
    rep  = ec[ec["tipo_evento"] == "reprogramacion_pool_saturado"]
    inv  = ec[ec["tipo_evento"] == "intervencion_nsp"]
    d = _cero()
    d["consulta"]             = int(ctl["costo_consulta_clp"].sum()) + diag
    d["traslado"]             = int(ctl["costo_viaje_clp"].sum())
    d["no_show"]              = int(ina["costo_clp"].sum()) - len(ina) * COSTO_ADMIN_REPROG_CLP
    d["reprog_pool_saturado"] = int(rep["costo_clp"].sum()) - len(rep) * COSTO_ADMIN_REPROG_CLP
    d["admin"]                = (len(ina) + len(rep)) * COSTO_ADMIN_REPROG_CLP
    d["intervencion_nsp"]     = int(inv["costo_clp"].sum())
    return d


_RECON = {
    "base":     _seed_base,
    "solo_ews": _seed_solo_ews,
    "solo_nsp": _seed_solo_nsp,
    "ews_nsp":  _seed_ews_nsp,
}

_GRUPOS = [
    ("consulta",            ["consulta"]),
    ("traslado",            ["traslado"]),
    ("inasistencia_reprog", ["no_show", "prof_no_disponible",
                             "reprog_pool_saturado", "admin", "inasistencia_post_nsp"]),
    ("intervencion_nsp",    ["intervencion_nsp"]),
]

_AHORRO = [
    ("consulta",                 ["consulta"]),
    ("traslado",                 ["traslado"]),
    ("inasistencia_paciente",    ["no_show"]),
    ("inasistencia_profesional", ["prof_no_disponible", "reprog_pool_saturado"]),
    ("administrativo",           ["admin"]),
    ("intervencion_nsp",         ["intervencion_nsp"]),
    ("inasistencia_post_nsp",    ["inasistencia_post_nsp"]),
]


def validar_y_promediar(seeds):
    filas_val = []
    acum = {esc: {c: [] for c in _COMP} for esc in _RECON}
    acum_total = {esc: [] for esc in _RECON}

    for esc, fn in _RECON.items():
        for s in seeds:
            comp  = fn(s)
            recon = sum(comp[c] for c in _COMP)
            auth  = _resumen_total(esc, s)
            dif   = recon - auth
            filas_val.append({"escenario": esc, "seed": s, "reconstruido": recon,
                              "autoritativo": auth, "diferencia_clp": dif})
            assert dif == 0, (
                f"RECONCILIACION FALLIDA {esc} seed {s}: "
                f"reconstruido={recon:,} vs autoritativo={auth:,} (dif={dif:,})"
            )
            for c in _COMP:
                acum[esc][c].append(comp[c])
            acum_total[esc].append(auth)

    filas_media = []
    for esc in _RECON:
        m   = {c: float(np.mean(acum[esc][c])) for c in _COMP}
        tot = float(np.mean(acum_total[esc]))
        fila = {"escenario": esc}
        fila.update({f"costo_{c}": int(round(m[c])) for c in _COMP})
        fila["costo_total"]  = int(round(tot))
        fila["pct_consulta"] = round(m["consulta"] / tot * 100, 1)
        fila["pct_traslado"] = round(m["traslado"] / tot * 100, 1)
        fila["n_replicas"]   = len(acum_total[esc])
        filas_media.append(fila)

    return pd.DataFrame(filas_media), pd.DataFrame(filas_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-replicas", type=int, default=30)
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.n_replicas)]
    DIR_OUT.mkdir(parents=True, exist_ok=True)

    df, df_val = validar_y_promediar(seeds)

    orden = {"base": 0, "solo_ews": 1, "solo_nsp": 2, "ews_nsp": 3}
    df = df.sort_values("escenario", key=lambda s: s.map(orden)).reset_index(drop=True)

    ruta_csv = DIR_OUT / "desglose_costos_por_escenario.csv"
    df.to_csv(ruta_csv, index=False)

    filas_res = []
    for _, f in df.iterrows():
        fila = {"escenario": f["escenario"]}
        for nombre, comps in _GRUPOS:
            fila[f"costo_{nombre}"] = int(sum(int(f[f"costo_{c}"]) for c in comps))
        fila["costo_total"] = int(f["costo_total"])
        filas_res.append(fila)
    df_res = pd.DataFrame(filas_res)
    ruta_res = DIR_OUT / "desglose_costos_resumen.csv"
    df_res.to_csv(ruta_res, index=False)

    base_row = df[df["escenario"] == "base"].iloc[0]
    filas_ah = []
    for _, f in df.iterrows():
        if f["escenario"] == "base":
            continue
        fila = {"escenario": f["escenario"]}
        suma = 0
        for nombre, comps in _AHORRO:
            v = int(sum(int(base_row[f"costo_{c}"]) - int(f[f"costo_{c}"]) for c in comps))
            fila[f"ahorro_{nombre}"] = v
            suma += v
        fila["ahorro_total"] = int(base_row["costo_total"]) - int(f["costo_total"])
        assert abs(suma - fila["ahorro_total"]) <= len(_COMP) + 1, (
            f"AHORRO no reconcilia {f['escenario']}: "
            f"suma componentes={suma:,} vs ahorro_total={fila['ahorro_total']:,}"
        )
        filas_ah.append(fila)
    df_ah = pd.DataFrame(filas_ah)
    ruta_ah = DIR_OUT / "desglose_ahorro_por_escenario.csv"
    df_ah.to_csv(ruta_ah, index=False)

    n_ok = (df_val["diferencia_clp"] == 0).sum()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=== RESUMEN ===")
    print(df_res.to_string(index=False))
    print("\n=== ATOMICO ===")
    print(df.to_string(index=False))
    print("\n=== AHORRO POR COMPONENTE (base - escenario) ===")
    print(df_ah.to_string(index=False))
    print(f"\nValidacion bit-a-bit: {n_ok}/{len(df_val)} exactas "
          f"(max |dif| = {df_val['diferencia_clp'].abs().max()} CLP)")
    print(f"CSV resumen : {ruta_res}")
    print(f"CSV atomico : {ruta_csv}")
    print(f"CSV ahorro  : {ruta_ah}")
