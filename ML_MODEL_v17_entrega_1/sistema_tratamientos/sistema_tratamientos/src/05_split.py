
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

FEATURES_MODELO = import_module("04_features").FEATURES_MODELO

logger = logging.getLogger(__name__)

RUTA_ENTRADA = Path(__file__).resolve().parent.parent / "data" / "processed" / "pacientes_con_features.csv"
DIR_SALIDA = Path(__file__).resolve().parent.parent / "data" / "processed"

def split_por_paciente(df: pd.DataFrame, seed: int = 42,
                       prop_train: float = 0.70,
                       prop_val: float = 0.15) -> tuple:
    rng = np.random.default_rng(seed=seed)

    pacientes = df["paciente_id"].unique()
    pacientes = rng.permutation(pacientes)

    n = len(pacientes)
    n_train = int(n * prop_train)
    n_val = int(n * prop_val)

    pacientes_train = pacientes[:n_train]
    pacientes_val = pacientes[n_train:n_train + n_val]
    pacientes_test = pacientes[n_train + n_val:]

    df_train = df[df["paciente_id"].isin(pacientes_train)].copy()
    df_val = df[df["paciente_id"].isin(pacientes_val)].copy()
    df_test = df[df["paciente_id"].isin(pacientes_test)].copy()

    return df_train, df_val, df_test

def aplicar_corte_temporal(df: pd.DataFrame, fraccion_final: float = 0.25) -> pd.DataFrame:
    df = df.sort_values(["paciente_id", "fecha_control"])
    resultado = []

    for _, grupo in df.groupby("paciente_id"):
        n = len(grupo)
        corte = int(n * (1 - fraccion_final))
        resultado.append(grupo.iloc[corte:])

    return pd.concat(resultado, ignore_index=True)

def separar_x_y(df: pd.DataFrame, columna_objetivo: str = "descompensacion_glicemica_90d"):
    df = df.copy()

    if "sexo" in df.columns and df["sexo"].dtype == object:
        df["sexo"] = (df["sexo"].astype(str).str.upper() == "M").astype(int)

    faltantes = [c for c in FEATURES_MODELO if c not in df.columns]
    if faltantes:
        logger.warning("separar_x_y: features faltantes %s", faltantes)
        raise ValueError(
            f"separar_x_y: faltan {len(faltantes)} features requeridas: {faltantes}"
        )

    y = df[columna_objetivo].astype(int)
    X = df[FEATURES_MODELO].copy()

    return X, y

def dividir_dataset(seed: int = 42):
    print("=" * 60)
    print(f"Leyendo: {RUTA_ENTRADA}")
    df = pd.read_csv(RUTA_ENTRADA)
    print(f"  shape: {df.shape}")
    print(f"  pacientes: {df['paciente_id'].nunique()}")

    print("\n1. Split por paciente (70/15/15)...")
    df_train, df_val, df_test = split_por_paciente(df, seed=seed)
    print(f"  train: {df_train['paciente_id'].nunique()} pacientes "
          f"({len(df_train)} filas)")
    print(f"  val:   {df_val['paciente_id'].nunique()} pacientes "
          f"({len(df_val)} filas)")
    print(f"  test:  {df_test['paciente_id'].nunique()} pacientes "
          f"({len(df_test)} filas)")

    print("\n2. Corte temporal en val y test (solo controles finales)...")
    df_val = aplicar_corte_temporal(df_val, fraccion_final=0.25)
    df_test = aplicar_corte_temporal(df_test, fraccion_final=0.25)
    print(f"  val filtrado:  {len(df_val)} filas (ultimo 25%)")
    print(f"  test filtrado: {len(df_test)} filas (ultimo 25%)")

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(DIR_SALIDA / "train.csv", index=False)
    df_val.to_csv(DIR_SALIDA / "val.csv", index=False)
    df_test.to_csv(DIR_SALIDA / "test.csv", index=False)

    print(f"\nArchivos guardados en: {DIR_SALIDA}")

    print("\n3. Validando distribucion del objetivo:")
    print(f"  train: {df_train['descompensacion_glicemica_90d'].mean():.2%}")
    print(f"  val:   {df_val['descompensacion_glicemica_90d'].mean():.2%}")
    print(f"  test:  {df_test['descompensacion_glicemica_90d'].mean():.2%}")

    print("\n4. Verificando que no hay leakage:")
    set_train = set(df_train["paciente_id"].unique())
    set_val = set(df_val["paciente_id"].unique())
    set_test = set(df_test["paciente_id"].unique())

    assert len(set_train & set_val) == 0, "LEAKAGE: pacientes compartidos train-val"
    assert len(set_train & set_test) == 0, "LEAKAGE: pacientes compartidos train-test"
    assert len(set_val & set_test) == 0, "LEAKAGE: pacientes compartidos val-test"
    print("  ok: ningun paciente compartido entre splits")

    return df_train, df_val, df_test

if __name__ == "__main__":
    dividir_dataset()
