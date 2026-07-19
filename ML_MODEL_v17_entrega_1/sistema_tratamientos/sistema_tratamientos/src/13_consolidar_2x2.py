import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

_DIR_BASE    = Path(__file__).resolve().parent.parent
_DIR_REPORTS = _DIR_BASE / "reports"
_DIR_OUT     = _DIR_REPORTS / "evaluacion_final"

_ESCENARIOS = ["base", "solo_ews", "solo_nsp", "ews_nsp"]

_COSTO_COLS = ("costo_total_clp", "costo_eventos_total_clp")

def _leer_costo(row: "pd.Series") -> float:
    for col in _COSTO_COLS:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    raise KeyError(
        f"No se encontro columna de costo en {list(row.index)}. "
        f"Esperado: {_COSTO_COLS}"
    )

def consolidar_2x2(
    n_replicas: int  = 30,
    seed: int        = 42,
    dir_reports: Path = _DIR_REPORTS,
    dir_out: Path    = _DIR_OUT,
) -> dict:
    dir_out.mkdir(parents=True, exist_ok=True)

    semillas = [seed + i for i in range(n_replicas)]

    faltantes = []
    for sem in semillas:
        for esc in _ESCENARIOS:
            p = dir_reports / f"simulacion_{esc}_resumen_seed{sem}.csv"
            if not p.exists():
                faltantes.append(str(p))
    if faltantes:
        raise ValueError(
            f"Faltan {len(faltantes)} archivo(s):\n" +
            "\n".join(f"  {f}" for f in faltantes)
        )

    costos: dict[str, list[float]] = {e: [] for e in _ESCENARIOS}
    for sem in semillas:
        for esc in _ESCENARIOS:
            row = pd.read_csv(
                dir_reports / f"simulacion_{esc}_resumen_seed{sem}.csv"
            ).iloc[0]
            costos[esc].append(_leer_costo(row))

    def _stats(vals: list[float]) -> dict:
        n    = len(vals)
        arr  = np.array(vals, dtype=float)
        mu   = float(arr.mean())
        sd   = float(arr.std(ddof=1)) if n > 1 else 0.0
        marg = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
        return {"media": mu, "sd": sd, "ic95_inf": mu - marg, "ic95_sup": mu + marg, "n": n}

    filas_costos = []
    for esc in _ESCENARIOS:
        sc = _stats(costos[esc])
        ahorros = [(b - c) / b for b, c in zip(costos["base"], costos[esc])] \
                  if esc != "base" else [0.0] * n_replicas
        sa = _stats(ahorros)
        filas_costos.append({
            "escenario":      esc,
            "costo_medio":    round(sc["media"]),
            "costo_sd":       round(sc["sd"]),
            "costo_ic95_inf": round(sc["ic95_inf"]),
            "costo_ic95_sup": round(sc["ic95_sup"]),
            "ahorro_medio":   round(sa["media"], 4),
            "ahorro_sd":      round(sa["sd"], 4),
            "ahorro_ic95_inf":round(sa["ic95_inf"], 4),
            "ahorro_ic95_sup":round(sa["ic95_sup"], 4),
            "n_replicas":     sc["n"],
        })

    df_costos = pd.DataFrame(filas_costos)
    df_costos.to_csv(dir_out / "comparacion_2x2_costos.csv", index=False)

    def _ahorro_series(alt):
        return [(b - a) / b for b, a in zip(costos["base"], alt)]

    a_ews  = _ahorro_series(costos["solo_ews"])
    a_nsp  = _ahorro_series(costos["solo_nsp"])
    a_comb = _ahorro_series(costos["ews_nsp"])
    suma_ind = [e + n for e, n in zip(a_ews, a_nsp)]
    subad    = [c - s for c, s in zip(a_comb, suma_ind)]

    series_map = [
        ("ahorro_ews",                 a_ews,    "ahorro por EWS aislado"),
        ("ahorro_nsp",                 a_nsp,    "ahorro por NSP aislado"),
        ("suma_individual_ews_nsp",    suma_ind, "suma si los efectos fueran aditivos"),
        ("ahorro_combinado_observado", a_comb,   "ahorro real del escenario combinado"),
        ("subaditividad",              subad,    "diferencia (negativo = sub-aditivo)"),
    ]

    filas_sub = []
    for nombre, serie, interp in series_map:
        s = _stats(serie)
        filas_sub.append({
            "metrica":        nombre,
            "media":          round(s["media"], 4),
            "sd":             round(s["sd"], 4),
            "ic95_inf":       round(s["ic95_inf"], 4),
            "ic95_sup":       round(s["ic95_sup"], 4),
            "n_replicas":     s["n"],
            "interpretacion": interp,
        })

    df_sub = pd.DataFrame(filas_sub)
    df_sub.to_csv(dir_out / "comparacion_2x2_subaditividad.csv", index=False)

    return {"costos": df_costos, "subaditividad": df_sub}

def _imprimir_resumen(resultados: dict, n: int, seed: int) -> None:
    dc = resultados["costos"].set_index("escenario")
    ds = resultados["subaditividad"].set_index("metrica")

    print(f"\nConsolidación 2x2 - {n} réplicas")
    print(f"\n{'Escenario':<12} {'Costo medio (CLP)':>22}  {'Ahorro vs base':>20}")
    print("-" * 60)
    for esc in _ESCENARIOS:
        r = dc.loc[esc]
        costo_s = f"${r['costo_medio']:,.0f} ± {r['costo_sd']:,.0f}"
        if esc == "base":
            ahorro_s = "0.0% ± 0.0%"
        else:
            ic_lo = r['ahorro_ic95_inf'] * 100
            ic_hi = r['ahorro_ic95_sup'] * 100
            ahorro_s = (f"{r['ahorro_medio']*100:.1f}% ± {r['ahorro_sd']*100:.1f}%"
                        f"  (IC95: {ic_lo:.1f}% - {ic_hi:.1f}%)")
        print(f"{esc:<12} {costo_s:>22}  {ahorro_s}")

    sub_media = ds.loc["subaditividad", "media"]
    sub_sd    = ds.loc["subaditividad", "sd"]
    sub_lo    = ds.loc["subaditividad", "ic95_inf"]
    sub_hi    = ds.loc["subaditividad", "ic95_sup"]
    if sub_media < -0.01:
        interp = "sub-aditivo (efectos solapados)"
    elif sub_media > 0.01:
        interp = "sinergico positivo"
    else:
        interp = "aditivo"

    print(f"\nSubaditividad: {sub_media*100:.2f}% ± {sub_sd*100:.2f}%"
          f"  (IC95: {sub_lo*100:.2f}% - {sub_hi*100:.2f}%)")
    print(f"Interpretación: {interp}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolida N replicas del diseno factorial 2x2 y calcula sub-aditividad."
    )
    parser.add_argument("--n-replicas",  type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--dir-reports", type=str, default=str(_DIR_REPORTS))
    parser.add_argument("--dir-out",     type=str, default=str(_DIR_OUT))
    args = parser.parse_args()

    res = consolidar_2x2(
        n_replicas  = args.n_replicas,
        seed        = args.seed,
        dir_reports = Path(args.dir_reports),
        dir_out     = Path(args.dir_out),
    )
    _imprimir_resumen(res, args.n_replicas, args.seed)
