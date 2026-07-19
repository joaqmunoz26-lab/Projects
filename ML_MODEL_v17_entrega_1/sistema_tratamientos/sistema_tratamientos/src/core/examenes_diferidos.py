from dataclasses import dataclass, field


@dataclass
class ExamenProgramado:
    nombre:                  str
    costo_clp:               int
    prioridad_clinica:       int
    obligatorio_diagnostico: bool
    mes_programado:          int
    estado:                  str             = "pendiente"

@dataclass
class CalendarioExamenes:
    examenes: list = field(default_factory=list)

    @classmethod
    def crear_para_paciente_ges(
        cls,
        examenes_recomendados: list,
    ) -> "CalendarioExamenes":

        programados = []
        for ex in examenes_recomendados:
            programados.append(ExamenProgramado(
                nombre=ex["nombre"],
                costo_clp=0,
                prioridad_clinica=ex["prioridad_clinica"],
                obligatorio_diagnostico=ex["obligatorio_diagnostico"],
                mes_programado=ex["mes_default"],
            ))
        return cls(examenes=programados)

    @classmethod
    def crear_para_paciente(
        cls,
        capacidad_pago_clp_mes: int,
        examenes_recomendados: list,
    ) -> "CalendarioExamenes":

        examenes_ordenados = sorted(
            examenes_recomendados,
            key=lambda e: (e["prioridad_clinica"], e["mes_default"]),
        )

        gasto_por_mes: dict = {}
        programados: list   = []

        for ex in examenes_ordenados:
            mes = ex["mes_default"]
            while True:
                gasto_acum = gasto_por_mes.get(mes, 0)
                if gasto_acum + ex["costo_clp"] <= capacidad_pago_clp_mes:
                    gasto_por_mes[mes] = gasto_acum + ex["costo_clp"]
                    programados.append(ExamenProgramado(
                        nombre=ex["nombre"],
                        costo_clp=ex["costo_clp"],
                        prioridad_clinica=ex["prioridad_clinica"],
                        obligatorio_diagnostico=ex["obligatorio_diagnostico"],
                        mes_programado=mes,
                    ))
                    break
                mes += 1
                if mes > 12:
                    programados.append(ExamenProgramado(
                        nombre=ex["nombre"],
                        costo_clp=ex["costo_clp"],
                        prioridad_clinica=ex["prioridad_clinica"],
                        obligatorio_diagnostico=ex["obligatorio_diagnostico"],
                        mes_programado=1,
                        estado="diferido",
                    ))
                    break

        return cls(examenes=programados)

    def examenes_mes(self, mes: int) -> list:
        return [e for e in self.examenes if e.mes_programado == mes]

    def examenes_obligatorios_mes_1(self) -> list:
        return [
            e for e in self.examenes
            if e.mes_programado == 1 and e.obligatorio_diagnostico
        ]

    @property
    def costo_total_clp(self) -> int:
        return sum(e.costo_clp for e in self.examenes)

    @property
    def meses_hasta_perfil_completo(self) -> int:
        if not self.examenes:
            return 0
        return max(e.mes_programado for e in self.examenes)

    @property
    def tiene_diferimiento(self) -> bool:
        return self.meses_hasta_perfil_completo > 1

if __name__ == "__main__":
    EXAMENES_DEMO = [
        {"nombre": "hba1c",          "costo_clp": 8000,  "prioridad_clinica": 1,
         "mes_default": 1, "obligatorio_diagnostico": True},
        {"nombre": "glicemia_ayunas","costo_clp": 4000,  "prioridad_clinica": 1,
         "mes_default": 1, "obligatorio_diagnostico": True},
        {"nombre": "perfil_lipidico","costo_clp": 12000, "prioridad_clinica": 2,
         "mes_default": 2, "obligatorio_diagnostico": False},
        {"nombre": "microalbuminuria","costo_clp": 8000,  "prioridad_clinica": 2,
         "mes_default": 2, "obligatorio_diagnostico": False},
        {"nombre": "funcion_renal_egfr","costo_clp": 6000, "prioridad_clinica": 3,
         "mes_default": 3, "obligatorio_diagnostico": False},
        {"nombre": "ecg_basal",      "costo_clp": 9000,  "prioridad_clinica": 3,
         "mes_default": 3, "obligatorio_diagnostico": False},
    ]

    for cap in [50000, 20000, 12000, 8000, 5000]:
        cal = CalendarioExamenes.crear_para_paciente(cap, EXAMENES_DEMO)
        obligs = cal.examenes_obligatorios_mes_1()
        print(f"\n  Capacidad ${cap:,}/mes:")
        print(f"    Meses hasta perfil completo : {cal.meses_hasta_perfil_completo}")
        print(f"    Obligatorios en mes 1       : {len(obligs)} "
              f"({[e.nombre for e in obligs]})")
        print(f"    Tiene diferimiento          : {cal.tiene_diferimiento}")
        print(f"    Costo total                 : ${cal.costo_total_clp:,}")
        for mes in range(1, cal.meses_hasta_perfil_completo + 1):
            names = [e.nombre for e in cal.examenes_mes(mes)]
            print(f"    Mes {mes}: {names}")
