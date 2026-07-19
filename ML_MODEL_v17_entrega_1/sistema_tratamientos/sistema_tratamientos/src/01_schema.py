from dataclasses import dataclass, field
from pathlib import Path

import yaml

RUTA_DICCIONARIO = Path(__file__).resolve().parent.parent / "00_data_dictionary.yaml"

@dataclass
class Variable:
    nombre: str
    tipo: str
    descripcion: str
    periodicidad: str
    unidad: str | None = None
    rango_min: float | None = None
    rango_max: float | None = None
    rango_normal_min: float | None = None
    rango_normal_max: float | None = None
    rango_objetivo_max: float | None = None
    rango_objetivo_min: float | None = None
    valores_permitidos: list | None = None
    fhir_path: str | None = None
    derivada: bool = False
    formula: str | None = None
    formato: str | None = None
    fuente: str | None = None

    def validar_valor(self, valor):
        if valor is None:
            return True, "ok"

        if self.valores_permitidos is not None and valor not in self.valores_permitidos:
            return False, f"{valor} no esta en {self.valores_permitidos}"

        if self.rango_min is not None and valor < self.rango_min:
            return False, f"{valor} < rango_min={self.rango_min}"

        if self.rango_max is not None and valor > self.rango_max:
            return False, f"{valor} > rango_max={self.rango_max}"

        return True, "ok"


@dataclass
class Schema:
    patologia: str
    poblacion_objetivo: dict
    variable_objetivo: dict
    variables: list = field(default_factory=list)

    def get_variable(self, nombre: str) -> Variable:
        for var in self.variables:
            if var.nombre == nombre:
                return var
        raise KeyError(f"Variable '{nombre}' no existe en el schema")

    def variables_numericas(self) -> list:
        return [v for v in self.variables if v.tipo in ("int", "float")]

    def variables_categoricas(self) -> list:
        return [v for v in self.variables if v.tipo == "categorica"]

    def variables_temporales(self) -> list:
        return [v for v in self.variables if v.periodicidad != "estatica"]


def cargar_schema(ruta: Path = RUTA_DICCIONARIO) -> Schema:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro diccionario en: {ruta}")

    with open(ruta, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    variables = [Variable(**v) for v in data["variables"]]

    return Schema(
        patologia=data["patologia"],
        poblacion_objetivo=data["poblacion_objetivo"],
        variable_objetivo=data["variable_objetivo"],
        variables=variables,
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Cargando schema del diccionario de datos...")
    print("=" * 60)

    schema = cargar_schema()

    print(f"\nPatologia: {schema.patologia}")
    print(f"Variable objetivo: {schema.variable_objetivo['nombre']}")
    print(f"Poblacion: {schema.poblacion_objetivo}")
    print(f"\nTotal de variables: {len(schema.variables)}")
    print(f"  - Numericas: {len(schema.variables_numericas())}")
    print(f"  - Categoricas: {len(schema.variables_categoricas())}")
    print(f"  - Temporales: {len(schema.variables_temporales())}")

    print("\nListado de variables:")
    for var in schema.variables:
        unidad = f" [{var.unidad}]" if var.unidad else ""
        print(f"  - {var.nombre}{unidad}: {var.descripcion}")

    print("\nValidaciones de ejemplo:")
    hba1c = schema.get_variable("hba1c")
    for valor in [6.5, 9.2, 20.0, -1.0]:
        ok, msg = hba1c.validar_valor(valor)
        estado = "OK" if ok else "ERROR"
        print(f"  hba1c={valor} -> [{estado}] {msg}")

    print("\nSchema cargado correctamente.")
