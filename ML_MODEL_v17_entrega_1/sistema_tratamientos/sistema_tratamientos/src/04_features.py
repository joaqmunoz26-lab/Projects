
from pathlib import Path

import pandas as pd

RUTA_ENTRADA = Path(__file__).resolve().parent.parent / "data" / "processed" / "pacientes_limpios.csv"
RUTA_SALIDA = Path(__file__).resolve().parent.parent / "data" / "processed" / "pacientes_con_features.csv"

FEATURES_MODELO = [
    "hba1c",
    "presion_sistolica",
    "presion_diastolica",
    "funcion_renal_egfr",
    "colesterol_ldl",
    "adherencia_tratamiento",
    "microalbuminuria",
    "ecg_anormal",
    "imc",
    "edad",
    "sexo",
    "hospitalizacion_previa_12m",
]

CLAVES_DATASET = [
    "paciente_id", "fecha_control", "control_num",
    "prob_descompensacion_90d", "descompensacion_glicemica_90d",
]

def generar_features(ruta_entrada: Path = RUTA_ENTRADA,
                     ruta_salida: Path = RUTA_SALIDA) -> pd.DataFrame:
    print("=" * 60)
    print(f"Leyendo: {ruta_entrada}")
    df = pd.read_csv(ruta_entrada)
    print(f"  shape inicial: {df.shape}")

    faltantes = [c for c in FEATURES_MODELO if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan {len(faltantes)} features requeridas en {ruta_entrada}: "
            f"{faltantes}"
        )

    claves = [c for c in CLAVES_DATASET if c in df.columns]
    df_out = df[claves + FEATURES_MODELO].copy()

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ruta_salida, index=False)

    print(f"\nFeatures generados: {ruta_salida}")
    print(f"  shape final: {df_out.shape}")
    print(f"  features del modelo: {len(FEATURES_MODELO)}")

    return df_out

if __name__ == "__main__":
    df = generar_features()
    print("\nFeatures del modelo (orden fijo):")
    for i, f in enumerate(FEATURES_MODELO, 1):
        print(f"  {i:2d}. {f}")
