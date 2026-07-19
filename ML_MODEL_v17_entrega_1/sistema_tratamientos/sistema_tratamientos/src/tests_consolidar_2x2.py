import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_cons = importlib.import_module("13_consolidar_2x2")
consolidar_2x2 = _cons.consolidar_2x2

def _escribir_replicas(rep: Path, seed: int, n: int,
                       costos: dict[str, list[float]]) -> None:
    for i in range(n):
        for esc, vals in costos.items():
            pd.DataFrame([{
                "escenario":      esc,
                "n_pacientes":    20,
                "costo_total_clp": vals[i],
                "total_presenciales": 10,
            }]).to_csv(
                rep / f"simulacion_{esc}_resumen_seed{seed + i}.csv",
                index=False,
            )

def test_consolidacion_2x2_replicas(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()
    out = tmp_path / "eval"

    _escribir_replicas(rep, seed=100, n=2, costos={
        "base":     [100_000, 110_000],
        "solo_ews": [ 50_000,  55_000],
        "solo_nsp": [ 90_000,  99_000],
        "ews_nsp":  [ 45_000,  49_500],
    })

    consolidar_2x2(n_replicas=2, seed=100, dir_reports=rep, dir_out=out)

    for fname in ("comparacion_2x2_costos.csv", "comparacion_2x2_subaditividad.csv"):
        assert (out / fname).exists(), f"{fname} no fue creado"

    dc = pd.read_csv(out / "comparacion_2x2_costos.csv")
    for col in ("escenario", "costo_medio", "costo_sd",
                "costo_ic95_inf", "costo_ic95_sup",
                "ahorro_medio", "ahorro_sd",
                "ahorro_ic95_inf", "ahorro_ic95_sup", "n_replicas"):
        assert col in dc.columns, f"Columna ausente en costos: {col}"

    assert (dc["n_replicas"] == 2).all()
    ahorro_base = dc[dc["escenario"] == "base"]["ahorro_medio"].iloc[0]
    assert ahorro_base == 0.0, f"Ahorro base debe ser 0.0, obtuvo {ahorro_base}"

    ds = pd.read_csv(out / "comparacion_2x2_subaditividad.csv")
    for col in ("metrica", "media", "sd", "ic95_inf", "ic95_sup", "n_replicas"):
        assert col in ds.columns, f"Columna ausente en subaditividad: {col}"

    assert (ds["n_replicas"] == 2).all()
    metricas_esperadas = {
        "ahorro_ews", "ahorro_nsp",
        "suma_individual_ews_nsp", "ahorro_combinado_observado", "subaditividad",
    }
    assert metricas_esperadas == set(ds["metrica"].tolist())

    sub = ds[ds["metrica"] == "subaditividad"]["media"].iloc[0]
    assert sub < 0, f"Sub-aditividad esperada negativa, obtuvo {sub}"

def test_consolidacion_aborta_si_falta_archivo(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()

    pd.DataFrame([{"escenario": "base", "costo_total_clp": 100_000, "n_pacientes": 20}
                  ]).to_csv(rep / "simulacion_base_resumen_seed42.csv", index=False)

    with pytest.raises(ValueError, match="simulacion_solo_ews"):
        consolidar_2x2(n_replicas=1, seed=42, dir_reports=rep, dir_out=tmp_path / "ev")
