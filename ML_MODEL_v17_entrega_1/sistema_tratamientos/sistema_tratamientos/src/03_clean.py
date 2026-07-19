
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_schema_mod = import_module("01_schema")

DIR_BASE = Path(__file__).resolve().parent.parent
RUTA_ENTRADA = DIR_BASE / "data" / "raw" / "pacientes_sinteticos.csv"
RUTA_SALIDA = DIR_BASE / "data" / "processed" / "pacientes_limpios.csv"
RUTA_REPORTE = DIR_BASE / "data" / "processed" / "reporte_faltantes.csv"

def reporte_faltantes_inicial(df: pd.DataFrame, schema) -> pd.DataFrame:
    filas = []
    for var in schema.variables:
        if var.nombre not in df.columns:
            continue
        n_total = len(df)
        n_faltantes = df[var.nombre].isna().sum()
        pct = 100 * n_faltantes / n_total if n_total > 0 else 0

        mecanismo = clasificar_mecanismo_faltante(df, var.nombre, schema)

        filas.append({
            "variable": var.nombre,
            "tipo": var.tipo,
            "n_faltantes": int(n_faltantes),
            "pct_faltantes": round(pct, 2),
            "mecanismo_sugerido": mecanismo,
            "accion_recomendada": sugerir_accion(var, mecanismo, pct),
        })

    return pd.DataFrame(filas).sort_values("pct_faltantes", ascending=False)

def clasificar_mecanismo_faltante(df, col, schema) -> str:
    if df[col].isna().sum() == 0:
        return "sin_faltantes"

    pct = df[col].isna().mean()

    variables_sensibles_mnar = [
        "funcion_renal_egfr", "colesterol_ldl", "hba1c",
    ]
    if col in variables_sensibles_mnar and pct > 0.10:
        return "posible_MNAR"

    if pct < 0.05:
        return "probable_MCAR"

    if "edad" in df.columns:
        mask_miss = df[col].isna()
        if mask_miss.sum() > 10 and (~mask_miss).sum() > 10:
            edad_miss = df.loc[mask_miss, "edad"].mean()
            edad_obs = df.loc[~mask_miss, "edad"].mean()
            if abs(edad_miss - edad_obs) > 3:
                return "probable_MAR"

    return "indeterminado"

def sugerir_accion(var, mecanismo: str, pct: float) -> str:
    if mecanismo == "sin_faltantes":
        return "ninguna"
    if pct > 40:
        return "considerar_descartar_variable"
    if mecanismo == "posible_MNAR":
        return "agregar_flag_missing + imputar_con_precaucion"
    if mecanismo == "probable_MCAR":
        return "imputar_mediana_o_moda"
    if mecanismo == "probable_MAR":
        return "imputar_knn_o_temporal"
    return "imputar_knn + flag_missing"

def agregar_flags_missing(df: pd.DataFrame, columnas: list,
                          min_pct: float = 0.01) -> pd.DataFrame:
    flags_creados = []
    for col in columnas:
        if col not in df.columns:
            continue
        pct = df[col].isna().mean()
        if pct >= min_pct:
            flag_name = f"{col}_was_missing"
            df[flag_name] = df[col].isna().astype(int)
            flags_creados.append(flag_name)

    if flags_creados:
        print(f"  flags creados: {len(flags_creados)} ({flags_creados[:3]}...)")
    return df

def detectar_outliers_con_flag(df: pd.DataFrame, schema) -> pd.DataFrame:
    for var in schema.variables_numericas():
        if var.nombre not in df.columns:
            continue

        mask_outlier = pd.Series(False, index=df.index)

        if var.rango_min is not None:
            mask_outlier |= df[var.nombre] < var.rango_min
        if var.rango_max is not None:
            mask_outlier |= df[var.nombre] > var.rango_max

        n_out = mask_outlier.sum()
        if n_out > 0:
            df[f"{var.nombre}_was_outlier"] = mask_outlier.astype(int)
            df.loc[mask_outlier, var.nombre] = np.nan

    return df

def imputar_temporal_por_paciente(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    df = df.sort_values(["paciente_id", "fecha_control"])

    for col in columnas:
        if col not in df.columns:
            continue
        df[col] = (df.groupby("paciente_id")[col]
                   .transform(lambda x: x.interpolate(method="linear")
                              .ffill().bfill()))

    return df

def preparar_para_split(df: pd.DataFrame, schema) -> pd.DataFrame:
    print("Preparando datos para split (sin leakage)...")

    antes = len(df)
    df = df.drop_duplicates(subset=["paciente_id", "fecha_control"], keep="first")
    if antes - len(df) > 0:
        print(f"  eliminados {antes - len(df)} duplicados")

    cols_temporales = [v.nombre for v in schema.variables_temporales()
                       if v.tipo in ("int", "float") and v.nombre in df.columns]
    df = agregar_flags_missing(df, cols_temporales)

    df = detectar_outliers_con_flag(df, schema)

    df = imputar_temporal_por_paciente(df, cols_temporales)

    df["fecha_control"] = pd.to_datetime(df["fecha_control"])
    df = df.sort_values(["paciente_id", "fecha_control"]).reset_index(drop=True)

    return df

def imputar_set(df_a_imputar: pd.DataFrame, df_referencia: pd.DataFrame,
                schema, metodo: str = "knn") -> pd.DataFrame:
    num_cols = [v.nombre for v in schema.variables_numericas()
                if v.nombre in df_a_imputar.columns]

    faltantes = df_a_imputar[num_cols].isna().sum().sum()
    if faltantes == 0:
        return df_a_imputar

    if metodo == "knn":
        imputador = KNNImputer(n_neighbors=5)
        imputador.fit(df_referencia[num_cols])
        df_a_imputar[num_cols] = imputador.transform(df_a_imputar[num_cols])
    elif metodo == "mediana":
        medianas = df_referencia[num_cols].median()
        df_a_imputar[num_cols] = df_a_imputar[num_cols].fillna(medianas)
    elif metodo == "media":
        medias = df_referencia[num_cols].mean()
        df_a_imputar[num_cols] = df_a_imputar[num_cols].fillna(medias)
    else:
        raise ValueError(f"Metodo de imputacion desconocido: {metodo}")

    cat_cols = [v.nombre for v in schema.variables_categoricas()
                if v.nombre in df_a_imputar.columns]
    for col in cat_cols:
        if df_a_imputar[col].isna().any():
            moda = df_referencia[col].mode()[0]
            df_a_imputar[col] = df_a_imputar[col].fillna(moda)

    return df_a_imputar

def convertir_tipos(df: pd.DataFrame, schema) -> pd.DataFrame:
    for var in schema.variables:
        if var.nombre not in df.columns:
            continue
        try:
            if var.tipo == "int":
                df[var.nombre] = df[var.nombre].round().astype("Int64")
            elif var.tipo == "float":
                df[var.nombre] = df[var.nombre].astype(float)
        except Exception:
            pass
    return df

def limpiar_dataset(ruta_entrada: Path = RUTA_ENTRADA,
                    ruta_salida: Path = RUTA_SALIDA) -> pd.DataFrame:
    print("=" * 60)
    print(f"Leyendo: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada)
    print(f"  shape inicial: {df.shape}")

    schema = _schema_mod.cargar_schema()

    print("\n1. Reporte diagnostico de faltantes (estado inicial)...")
    reporte = reporte_faltantes_inicial(df, schema)
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    reporte.to_csv(RUTA_REPORTE, index=False)
    print(reporte.head(10).to_string(index=False))

    print("\n2. Aplicando preparacion pre-split...")
    df = preparar_para_split(df, schema)

    print("\n3. Convirtiendo tipos...")
    df = convertir_tipos(df, schema)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)

    print(f"\nDataset limpio guardado en: {ruta_salida}")
    print(f"Reporte de faltantes en: {RUTA_REPORTE}")
    print(f"  shape final: {df.shape}")
    print(f"  columnas nuevas (flags): "
          f"{[c for c in df.columns if '_was_' in c][:5]}...")

    print("\nNota metodologica:")
    print("  La imputacion final (KNN o mediana) se debe ejecutar DESPUES del")
    print("  split, usando imputar_set() importada desde 05_split.py.")
    print("  Esto evita data leakage entre train/val/test.")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputador", type=str, default="knn",
                        choices=["knn", "mediana", "media"])
    args = parser.parse_args()
    limpiar_dataset()
