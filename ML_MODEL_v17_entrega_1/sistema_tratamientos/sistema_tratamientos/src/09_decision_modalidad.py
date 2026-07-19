
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAIZ      = Path(__file__).resolve().parent.parent
RUTA_YAML = RAIZ / "00_clases_modalidad.yaml"
CLASES_ESPERADAS = {
    "PRESENCIAL_URGENTE", "PRESENCIAL_CONTROL",
    "TELEMEDICINA_SINCRONICA", "REVISION_ASINCRONA", "CONTINUAR_TRATAMIENTO",
}
ALIAS = {"adherencia": "adherencia_tratamiento"}

def cargar_yaml(ruta: Path = RUTA_YAML) -> dict:
    with open(ruta, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    clases = set(data.get("clases", {}).keys())
    if faltantes := CLASES_ESPERADAS - clases:
        raise ValueError(f"Clases faltantes en YAML: {faltantes}")
    for r in data.get("reglas", []):
        for campo in ("id", "fase", "condiciones", "decision", "prioridad_regla"):
            if campo not in r:
                raise ValueError(f"Regla mal formada (falta '{campo}'): {r}")
        if r["decision"] not in clases:
            raise ValueError(f"Regla {r['id']} referencia clase inexistente: {r['decision']}")
    data["reglas"] = sorted(data["reglas"], key=lambda r: r["prioridad_regla"])
    return data

def _resolver(var: str, datos: dict, riesgo: float):
    if var == "riesgo_modelo":
        return riesgo
    if var in datos:
        return datos[var]
    return datos.get(ALIAS[var]) if var in ALIAS else None

def evaluar_condiciones(condiciones: dict, datos: dict, riesgo: float,
                        evento: str = None) -> bool:
    if not condiciones:
        return True
    for key, val in condiciones.items():
        if key == "evento":
            if evento != val:
                return False
        elif key.endswith("_ultimos_dias"):
            base = key[: -len("_ultimos_dias")]
            if datos.get(f"dias_desde_{base}", float("inf")) > val:
                return False
        elif key.endswith("_min"):
            dato = _resolver(key[:-4], datos, riesgo)
            if dato is None or dato < val:
                return False
        elif key.endswith("_max"):
            dato = _resolver(key[:-4], datos, riesgo)
            if dato is None or dato > val:
                return False
        elif key.endswith("_range"):
            dato = _resolver(key[:-6], datos, riesgo)
            if dato is None or not (val[0] <= dato <= val[1]):
                return False
        elif isinstance(val, bool):
            if datos.get(key, False) != val:
                return False
        elif isinstance(val, str):
            if datos.get(key) != val:
                return False
    return True

def _aplicar_reglas(datos: dict, riesgo: float, fase: str,
                    evento: str, yaml_data: dict) -> dict:
    reglas_fase = [r for r in yaml_data["reglas"] if r["fase"] == fase]
    if not reglas_fase:
        raise ValueError(f"No hay reglas para la fase '{fase}' en el YAML.")
    regla = next(
        (r for r in reglas_fase
         if evaluar_condiciones(r["condiciones"], datos, riesgo, evento)),
        None,
    )
    if regla is None:
        raise ValueError(
            f"Ninguna regla matcheó (fase='{fase}', riesgo={riesgo:.2f}). "
            "Agregá una regla default en el YAML."
        )
    nombre_clase = regla["decision"]
    ci = yaml_data["clases"][nombre_clase]
    return {
        "clase_modalidad":        ci["codigo"],
        "nombre_modalidad":       nombre_clase,
        "regla_aplicada":         regla["id"],
        "nombre_regla":           regla["nombre"],
        "justificacion":          (f"Riesgo={riesgo:.2f} en fase '{fase}'. "
                                   f"Regla: {regla['nombre']}."),
        "plazo_recomendado_dias": ci["plazo_max_dias"],
        "modalidad":              ci["modalidad"],
        "prioridad":              ci["prioridad"],
        "factores_principales":   [],
    }

_YAML_DATA = cargar_yaml()

def decidir_modalidad_desde_riesgo(riesgo: float, datos_paciente: dict,
                                    fase: str, evento: str = None) -> dict:
    return _aplicar_reglas(datos_paciente, riesgo, fase, evento, _YAML_DATA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor de decisión de modalidad de atención")
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--riesgo",     type=float)
    parser.add_argument("--fase",       choices=["diagnostico", "tratamiento", "alta"],
                        default="tratamiento")
    parser.add_argument("--hba1c",      type=float, default=7.0)
    parser.add_argument("--adherencia", type=float, default=0.80)
    args = parser.parse_args()

    if args.test:
        casos = [
            ("Descompensación grave",        0.85, "tratamiento", None,
             {"hba1c": 9.8, "adherencia_tratamiento": 0.40, "sintomas_activos": True}),
            ("Primera consulta",             0.30, "diagnostico", None,
             {"es_primera_consulta": True}),
            ("Moderado con datos remotos",   0.52, "tratamiento", None,
             {"hba1c": 7.8, "adherencia_tratamiento": 0.75, "datos_remotos_disponibles": True}),
            ("Solo examen de seguimiento",   0.35, "tratamiento", "examenes_seguimiento",
             {"hba1c": 7.2, "adherencia_tratamiento": 0.80}),
            ("Paciente estable y adherente", 0.18, "tratamiento", None,
             {"hba1c": 6.8, "adherencia_tratamiento": 0.88, "datos_remotos_disponibles": True}),
        ]
        print("=" * 70)
        print("CASOS DE PRUEBA - 09_decision_modalidad.py")
        print("=" * 70)
        for i, (desc, riesgo, fase, evento, datos) in enumerate(casos, 1):
            r = decidir_modalidad_desde_riesgo(riesgo, datos, fase, evento)
            entrada = f"riesgo={riesgo} | fase={fase}" + (f" | evento={evento}" if evento else "")
            print(f"\nCaso {i}: {desc}")
            print(f"  Entrada   : {entrada}")
            print(f"  Modalidad : [{r['clase_modalidad']}] {r['nombre_modalidad']}")
            print(f"  Regla     : {r['regla_aplicada']} - {r['nombre_regla']}")
            print(f"  Plazo     : {r['plazo_recomendado_dias']} días | {r['prioridad']}")

    elif args.riesgo is not None:
        import json
        datos = {"hba1c": args.hba1c, "adherencia_tratamiento": args.adherencia}
        print(json.dumps(
            decidir_modalidad_desde_riesgo(args.riesgo, datos, args.fase),
            ensure_ascii=False, indent=2,
        ))
    else:
        parser.print_help()
