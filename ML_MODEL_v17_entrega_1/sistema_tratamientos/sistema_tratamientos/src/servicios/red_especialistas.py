import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.perfiles import ZonaGeografica

PRESTAMOS_DESDE_ARRIBA = {
    "RURAL_CERCANO": 1,
    "RURAL_AISLADO": 0,
}

CAPACIDAD_DIARIA_POR_ESPECIALISTA = 1

@dataclass
class Especialista:
    id:                    str
    especialidad:          str
    zonas_cobertura:       set[ZonaGeografica]
    capacidad_diaria:      int
    carga_actual_consultas: int = 0
    disponible:            bool = True

    @property
    def carga_pct(self) -> float:
        return (self.carga_actual_consultas / self.capacidad_diaria) * 100

    @property
    def puede_aceptar_caso(self) -> bool:
        return self.disponible and self.carga_pct < 100

class RedEspecialistas:
    def __init__(self):
        self._especialistas: list[Especialista] = self._cargar_mvp()
        self._prestamos_usados_hoy: dict = {"RURAL_CERCANO": 0}

    def asignar_con_cascade(
        self,
        zona_paciente: ZonaGeografica,
    ) -> tuple:
        zona_val = zona_paciente.value.upper()

        nativos = [
            e for e in self._especialistas
            if zona_paciente in e.zonas_cobertura and e.puede_aceptar_caso
        ]
        if nativos:
            candidato = min(nativos, key=lambda e: e.carga_pct)
            candidato.carga_actual_consultas += 1
            return ("nativo", candidato.id)

        max_prestamos = PRESTAMOS_DESDE_ARRIBA.get(zona_val, 0)
        if max_prestamos > 0:
            usados = self._prestamos_usados_hoy.get(zona_val, 0)
            if usados < max_prestamos:
                zona_sup = ZonaGeografica.URBANO
                prestados = [
                    e for e in self._especialistas
                    if zona_sup in e.zonas_cobertura and e.puede_aceptar_caso
                ]
                if prestados:
                    candidato = min(prestados, key=lambda e: e.carga_pct)
                    candidato.carga_actual_consultas += 1
                    self._prestamos_usados_hoy[zona_val] = usados + 1
                    return ("prestamo", candidato.id)

        return ("saturado", None)

    def resetear_prestamos_diarios(self) -> None:
        for k in self._prestamos_usados_hoy:
            self._prestamos_usados_hoy[k] = 0

    def resetear_cargas_diarias(self) -> None:
        for e in self._especialistas:
            e.carga_actual_consultas = 0
            e.disponible = True
        self.resetear_prestamos_diarios()

    def buscar_por_id(self, especialista_id: str) -> Especialista | None:
        return next((e for e in self._especialistas if e.id == especialista_id), None)

    def listar_todos(self) -> list[Especialista]:
        return list(self._especialistas)

    def buscar_respaldo_mas_cercano(
        self,
        especialidad:    str,
        zona_paciente:   ZonaGeografica,
        carga_maxima_pct: float = 80.0,
        excluir_ids:     set[str] | None = None,
    ) -> Especialista | None:
        excluidos = excluir_ids or set()
        candidatos = [
            e for e in self._especialistas
            if e.especialidad == especialidad
            and zona_paciente in e.zonas_cobertura
            and e.disponible
            and e.carga_pct < carga_maxima_pct
            and e.id not in excluidos
        ]
        if not candidatos:
            return None
        return min(candidatos, key=lambda e: e.carga_pct)

    def asignar_caso(self, especialista_id: str) -> bool:
        e = self.buscar_por_id(especialista_id)
        if e is None or not e.puede_aceptar_caso:
            return False
        e.carga_actual_consultas += 1
        return True

    def liberar_caso(self, especialista_id: str) -> bool:
        e = self.buscar_por_id(especialista_id)
        if e is None:
            return False
        e.carga_actual_consultas = max(0, e.carga_actual_consultas - 1)
        return True

    def notificar_feedback_historial(
        self,
        especialista_id: str,
        paciente_id:     str,
        resultado:       str,
    ) -> None:
        logging.debug(f"feedback {especialista_id} | paciente={paciente_id} | resultado={resultado}")

    def esta_disponible(self, especialista_id: str) -> bool:
        esp = self.buscar_por_id(especialista_id)
        return esp is not None and esp.puede_aceptar_caso

    def obtener_especialidad(self, especialista_id: str) -> str | None:
        esp = self.buscar_por_id(especialista_id)
        return esp.especialidad if esp else None

    def estadisticas_red(self) -> dict:
        total       = len(self._especialistas)
        disponibles = sum(1 for e in self._especialistas if e.disponible)
        saturados   = sum(1 for e in self._especialistas if e.carga_pct >= 100)
        cargas      = [e.carga_pct for e in self._especialistas]
        carga_prom  = round(sum(cargas) / total, 1) if total else 0.0

        cobertura_zona = {}
        for zona in ZonaGeografica:
            cobertura_zona[zona.value] = sum(
                1 for e in self._especialistas if zona in e.zonas_cobertura
            )

        cobertura_esp = {}
        for e in self._especialistas:
            cobertura_esp[e.especialidad] = cobertura_esp.get(e.especialidad, 0) + 1

        return {
            "total":              total,
            "disponibles":        disponibles,
            "saturados":          saturados,
            "carga_promedio_pct": carga_prom,
            "cobertura_por_zona": cobertura_zona,
            "cobertura_por_especialidad": cobertura_esp,
        }

    @staticmethod
    def _cargar_mvp() -> list[Especialista]:
        U  = ZonaGeografica.URBANO
        RC = ZonaGeografica.RURAL_CERCANO
        RA = ZonaGeografica.RURAL_AISLADO
        CAP = CAPACIDAD_DIARIA_POR_ESPECIALISTA

        urbanos = [
            Especialista(
                id=f"doc_{i:03d}",
                especialidad="diabetologia",
                zonas_cobertura={U},
                capacidad_diaria=CAP,
            )
            for i in range(1, 7)
        ]

        rurales_cercanos = [
            Especialista(
                id=f"doc_{i:03d}",
                especialidad="diabetologia",
                zonas_cobertura={RC},
                capacidad_diaria=CAP,
            )
            for i in range(7, 10)
        ]

        rural_aislado = [
            Especialista(
                id="doc_010",
                especialidad="diabetologia",
                zonas_cobertura={RA},
                capacidad_diaria=CAP,
            )
        ]

        return urbanos + rurales_cercanos + rural_aislado

if __name__ == "__main__":
    red = RedEspecialistas()

    print("10 especialistas, cap = 1 consulta/día")
    for e in red.listar_todos():
        zonas = ", ".join(z.value for z in sorted(e.zonas_cobertura, key=lambda z: z.value))
        print(f"  {e.id:<10} esp={e.especialidad:<15} cap={e.capacidad_diaria} zonas=[{zonas}]")

    print()
    stats = red.estadisticas_red()
    print(f"Estadisticas: total={stats['total']} disponibles={stats['disponibles']} "
          f"saturados={stats['saturados']} carga_prom={stats['carga_promedio_pct']}%")
    print(f"Cobertura por zona: {stats['cobertura_por_zona']}")

    print()
    print("-- Cascade RURAL_CERCANO (pool nativo disponible) --")
    tipo, eid = red.asignar_con_cascade(ZonaGeografica.RURAL_CERCANO)
    print(f"  Tipo={tipo}  especialista={eid}")

    print("-- Saturar pool RURAL_CERCANO (doc_007-009) --")
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo2, eid2 = red.asignar_con_cascade(ZonaGeografica.RURAL_CERCANO)
    print(f"  Tipo={tipo2}  especialista={eid2}  (préstamo de URBANO)")

    print("-- Saturar RURAL_AISLADO (doc_010) --")
    red.buscar_por_id("doc_010").carga_actual_consultas = 1
    tipo3, eid3 = red.asignar_con_cascade(ZonaGeografica.RURAL_AISLADO)
    print(f"  Tipo={tipo3}  especialista={eid3}  (sin préstamo posible)")
