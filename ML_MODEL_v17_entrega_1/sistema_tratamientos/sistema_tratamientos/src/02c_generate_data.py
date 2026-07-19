
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_distr = import_module("02a_distributions")
_rules = import_module("02b_clinical_rules")

DISTRIBUCIONES = _distr.DISTRIBUCIONES

RUTA_SALIDA = Path(__file__).resolve().parent.parent / "data" / "raw" / "pacientes_sinteticos.csv"

def muestra_truncnorm(rng, media, sd, minimo, maximo):
    for _ in range(50):
        v = rng.normal(media, sd)
        if minimo <= v <= maximo:
            return v
    return float(np.clip(rng.normal(media, sd), minimo, maximo))

def generar_paciente_basal(paciente_id: int, rng) -> dict:
    sexo = rng.choice(["M", "F"], p=[0.47, 0.53])
    edad = muestra_truncnorm(rng, 60.0, 10.0, 40, 80)

    if sexo == "M":
        talla = muestra_truncnorm(rng, 171, 7, 140, 210)
    else:
        talla = muestra_truncnorm(rng, 158, 6, 140, 210)

    imc = muestra_truncnorm(rng, 30.5, 5.2, 18, 50)
    peso = imc * (talla / 100) ** 2

    hba1c = muestra_truncnorm(rng, 7.4, 1.3, 6.5, 14.0)
    glucosa = 28.7 * hba1c - 46.7 + rng.normal(0, 20)
    glucosa = float(np.clip(glucosa, 60, 400))

    presion_sis = muestra_truncnorm(rng, 135, 15, 90, 200)
    presion_dia = muestra_truncnorm(rng, 82, 10, 50, 120)
    ldl = muestra_truncnorm(rng, 110, 32, 40, 250)

    egfr = 120 - 0.8 * (edad - 40) + rng.normal(0, 15)
    egfr = float(np.clip(egfr, 15, 120))

    adherencia = float(np.clip(rng.beta(6.0, 2.0), 0.0, 1.0))
    num_med = max(1, min(12, rng.poisson(4.5)))
    hosp = int(rng.random() < 0.08)
    anios_dx = min(40, rng.exponential(8.0))

    microalb = rng.lognormal(_distr.MICROALBUMINURIA_BASAL_MEANLOG,
                             _distr.MICROALBUMINURIA_BASAL_SIGMALOG)
    if hba1c > 8.0:
        microalb *= 1.3
    if egfr < 60:
        microalb *= 1.4
    microalb = float(np.clip(microalb, 1.0, 3000.0))

    p_ecg = _distr.ECG_ANORMAL_BASAL_P
    if edad > 70:
        p_ecg += 0.10
    if presion_sis > 150:
        p_ecg += 0.08
    if anios_dx > 10:
        p_ecg += 0.05
    ecg_anormal = int(rng.random() < min(p_ecg, 0.85))

    return {
        "paciente_id": f"P{paciente_id:05d}",
        "edad": int(edad),
        "sexo": sexo,
        "talla_cm": round(talla, 1),
        "anios_diagnostico": round(anios_dx, 1),
        "hba1c": round(hba1c, 2),
        "glucosa_ayunas": round(glucosa, 1),
        "presion_sistolica": int(presion_sis),
        "presion_diastolica": int(presion_dia),
        "peso_kg": round(peso, 1),
        "colesterol_ldl": round(ldl, 1),
        "funcion_renal_egfr": round(egfr, 1),
        "adherencia_tratamiento": round(adherencia, 3),
        "num_medicamentos": int(num_med),
        "hospitalizacion_previa_12m": hosp,
        "microalbuminuria": round(microalb, 1),
        "ecg_anormal": ecg_anormal,
    }

def generar_trayectoria(basal: dict, n_controles: int, rng) -> list:
    registros = []

    fecha_inicial = datetime(2022, 1, 1) + timedelta(days=int(rng.integers(0, 365)))

    estado = basal.copy()
    trayectoria_hba1c = [estado["hba1c"]]

    for t in range(n_controles):
        fecha = fecha_inicial + timedelta(days=90 * t)

        if t > 0:
            estado["hba1c"] = _rules.evolucionar_hba1c(
                estado["hba1c"], estado["adherencia_tratamiento"],
                estado["num_medicamentos"], rng
            )
            estado["adherencia_tratamiento"] = _rules.evolucionar_adherencia(
                estado["adherencia_tratamiento"], estado["edad"],
                estado["num_medicamentos"], rng
            )
            estado["funcion_renal_egfr"] = _rules.evolucionar_egfr(
                estado["funcion_renal_egfr"], estado["hba1c"],
                estado["presion_sistolica"], estado["edad"], rng
            )
            estado["peso_kg"] = _rules.evolucionar_peso(
                estado["peso_kg"], estado["talla_cm"], rng
            )
            imc = estado["peso_kg"] / (estado["talla_cm"] / 100) ** 2
            estado["presion_sistolica"] = _rules.evolucionar_presion(
                estado["presion_sistolica"], estado["adherencia_tratamiento"],
                imc, rng, es_sistolica=True
            )
            estado["presion_diastolica"] = _rules.evolucionar_presion(
                estado["presion_diastolica"], estado["adherencia_tratamiento"],
                imc, rng, es_sistolica=False
            )
            estado["glucosa_ayunas"] = round(
                28.7 * estado["hba1c"] - 46.7 + rng.normal(0, 20), 1
            )
            estado["glucosa_ayunas"] = float(np.clip(estado["glucosa_ayunas"], 60, 400))

            estado["colesterol_ldl"] = round(_rules.evolucionar_ldl(
                estado["colesterol_ldl"], estado["adherencia_tratamiento"],
                estado["edad"], rng
            ), 1)
            estado["microalbuminuria"] = round(_rules.evolucionar_microalbuminuria(
                estado["microalbuminuria"], estado["hba1c"],
                estado["presion_sistolica"], estado["adherencia_tratamiento"], rng
            ), 1)
            estado["ecg_anormal"] = _rules.evolucionar_ecg(
                estado["ecg_anormal"], estado["edad"], estado["hba1c"],
                estado["presion_sistolica"],
                bool(estado["hospitalizacion_previa_12m"]), rng
            )

            trayectoria_hba1c.append(estado["hba1c"])

        imc_actual = estado["peso_kg"] / (estado["talla_cm"] / 100) ** 2

        prob = _rules.probabilidad_descompensacion_90d(
            trayectoria_hba1c,
            estado["adherencia_tratamiento"],
            estado["hba1c"],
            estado["funcion_renal_egfr"],
            bool(estado["hospitalizacion_previa_12m"]),
            microalbuminuria=estado["microalbuminuria"],
            ecg_anormal=estado["ecg_anormal"],
        )
        evento = _rules.generar_evento_descompensacion(prob, rng)

        registro = {
            **estado,
            "fecha_control": fecha.strftime("%Y-%m-%d"),
            "control_num": t + 1,
            "imc": round(imc_actual, 2),
            "prob_descompensacion_90d": round(prob, 4),
            "descompensacion_glicemica_90d": evento,
        }
        registros.append(registro)

    return registros

def generar_dataset(n_pacientes: int = 5000, n_controles: int = 8,
                    seed: int = 42) -> pd.DataFrame:
    print(f"Generando {n_pacientes} pacientes con {n_controles} controles cada uno...")
    rng = np.random.default_rng(seed=seed)

    todos_registros = []
    for pid in range(n_pacientes):
        if (pid + 1) % 500 == 0:
            print(f"  procesados {pid+1}/{n_pacientes}")

        basal = generar_paciente_basal(pid, rng)
        registros = generar_trayectoria(basal, n_controles, rng)
        todos_registros.extend(registros)

    return pd.DataFrame(todos_registros)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Numero de pacientes")
    parser.add_argument("--controles", type=int, default=8, help="Controles por paciente")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    df = generar_dataset(args.n, args.controles, args.seed)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False)

    print(f"\nDataset generado: {RUTA_SALIDA}")
    print(f"Filas: {len(df)}")
    print(f"Pacientes unicos: {df['paciente_id'].nunique()}")
    print(f"Tasa evento (descompensacion): {df['descompensacion_glicemica_90d'].mean():.2%}")
    print("\nResumen estadistico:")
    print(df[["hba1c", "adherencia_tratamiento", "funcion_renal_egfr"]].describe().round(2))
