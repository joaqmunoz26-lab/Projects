from pathlib import Path
import pandas as pd

RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "processed" / "pacientes_con_features.csv"

def cargar_historial_paciente(paciente_id: str,
                              ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    historial = df[df["paciente_id"] == paciente_id].copy()
    historial = historial.sort_values("fecha_control").reset_index(drop=True)
    return historial

def cargar_cohorte_completa(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    return pd.read_csv(ruta)

def listar_pacientes(ruta: Path = RUTA_DATOS) -> list:
    df = pd.read_csv(ruta)
    return sorted(df["paciente_id"].unique().tolist())

def metadatos_paciente(paciente_id: str, ruta: Path = RUTA_DATOS) -> dict:
    historial = cargar_historial_paciente(paciente_id, ruta)
    if len(historial) == 0:
        return None

    primer = historial.iloc[0]
    return {
        "paciente_id": paciente_id,
        "edad": int(primer["edad"]),
        "sexo": str(primer["sexo"]),
        "anios_diagnostico": float(primer["anios_diagnostico"]),
        "total_controles": len(historial),
        "primer_control": str(historial["fecha_control"].iloc[0]),
        "ultimo_control": str(historial["fecha_control"].iloc[-1]),
    }

if __name__ == "__main__":
    print("Probando loader sintetico...")

    pacientes = listar_pacientes()
    print(f"Total pacientes disponibles: {len(pacientes)}")
    print(f"Primeros 5: {pacientes[:5]}")

    if pacientes:
        pid = pacientes[0]
        meta = metadatos_paciente(pid)
        print(f"\nMetadatos de {pid}: {meta}")

        historial = cargar_historial_paciente(pid)
        print(f"\nHistorial ({len(historial)} controles):")
        cols = ["fecha_control", "hba1c", "adherencia_tratamiento",
                "presion_sistolica", "funcion_renal_egfr"]
        print(historial[cols].to_string(index=False))
