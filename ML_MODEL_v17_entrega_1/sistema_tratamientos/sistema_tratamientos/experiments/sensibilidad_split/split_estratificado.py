from pathlib import Path
import pandas as pd
import numpy as np

RUTA_ENTRADA = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "pacientes_con_features.csv"
)
DIR_SALIDA = Path(__file__).resolve().parent / "data"

def split_estratificado(
    df: pd.DataFrame,
    seed: int = 42,
    prop_train: float = 0.70,
    prop_val: float = 0.15,
) -> tuple:

    rng = np.random.default_rng(seed=seed)

    etiqueta = (
        df.groupby("paciente_id")["descompensacion_glicemica_90d"]
        .max()
        .astype(int)
        .rename("estrato")
    )

    pacientes_pos = etiqueta[etiqueta == 1].index.values
    pacientes_neg = etiqueta[etiqueta == 0].index.values

    def _cortar(pacs, p_train, p_val):
        n = len(pacs)
        n_train = int(n * p_train)
        n_val = int(n * p_val)
        return pacs[:n_train], pacs[n_train:n_train + n_val], pacs[n_train + n_val:]

    pos_train, pos_val, pos_test = _cortar(
        rng.permutation(pacientes_pos), prop_train, prop_val
    )
    neg_train, neg_val, neg_test = _cortar(
        rng.permutation(pacientes_neg), prop_train, prop_val
    )

    pacientes_train = np.concatenate([pos_train, neg_train])
    pacientes_val   = np.concatenate([pos_val,   neg_val])
    pacientes_test  = np.concatenate([pos_test,  neg_test])

    df_train = df[df["paciente_id"].isin(pacientes_train)].copy()
    df_val   = df[df["paciente_id"].isin(pacientes_val)].copy()
    df_test  = df[df["paciente_id"].isin(pacientes_test)].copy()

    return df_train, df_val, df_test

def aplicar_corte_temporal(
    df: pd.DataFrame, fraccion_final: float = 0.25
) -> pd.DataFrame:
    df = df.sort_values(["paciente_id", "fecha_control"])
    resultado = []
    for _pid, grupo in df.groupby("paciente_id"):
        n = len(grupo)
        corte = int(n * (1 - fraccion_final))
        resultado.append(grupo.iloc[corte:])
    return pd.concat(resultado, ignore_index=True)

def generar_splits(seed: int = 42) -> tuple:
    print("=" * 60)
    print("Experimento: Split estratificado (sensibilidad_split)")
    print("=" * 60)
    print(f"Leyendo: {RUTA_ENTRADA}")
    df = pd.read_csv(RUTA_ENTRADA)
    print(f"  shape: {df.shape} | pacientes: {df['paciente_id'].nunique()}")
    print(f"  prevalencia global: {df['descompensacion_glicemica_90d'].mean():.4f}")

    print("\n[1] Split estratificado por paciente (70/15/15)...")
    df_train, df_val, df_test = split_estratificado(df, seed=seed)
    for nombre, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        prev = part["descompensacion_glicemica_90d"].mean()
        print(
            f"  {nombre}: {part['paciente_id'].nunique()} pac | "
            f"{len(part)} filas | prevalencia {prev:.4f}"
        )

    print("\n[2] Corte temporal en val y test (ultimo 25% de controles)...")
    df_val  = aplicar_corte_temporal(df_val,  fraccion_final=0.25)
    df_test = aplicar_corte_temporal(df_test, fraccion_final=0.25)
    for nombre, part in [("val", df_val), ("test", df_test)]:
        prev = part["descompensacion_glicemica_90d"].mean()
        print(f"  {nombre} filtrado: {len(part)} filas | prevalencia {prev:.4f}")

    print("\n[3] Verificando leakage...")
    set_train = set(df_train["paciente_id"].unique())
    set_val   = set(df_val["paciente_id"].unique())
    set_test  = set(df_test["paciente_id"].unique())
    assert len(set_train & set_val)  == 0, "LEAKAGE train-val"
    assert len(set_train & set_test) == 0, "LEAKAGE train-test"
    assert len(set_val  & set_test)  == 0, "LEAKAGE val-test"
    print("ningún paciente compartido entre splits")

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(DIR_SALIDA / "train_estratificado.csv", index=False)
    df_val.to_csv(DIR_SALIDA   / "val_estratificado.csv",   index=False)
    df_test.to_csv(DIR_SALIDA  / "test_estratificado.csv",  index=False)
    print(f"\nArchivos guardados en: {DIR_SALIDA}")

    return df_train, df_val, df_test

if __name__ == "__main__":
    generar_splits()
