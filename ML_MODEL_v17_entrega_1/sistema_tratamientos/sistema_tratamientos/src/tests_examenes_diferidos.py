import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.examenes_diferidos import CalendarioExamenes
from pipeline.fase1_ingesta import EXAMENES_DIAGNOSTICOS_GES_DM2

EXAMENES_DM2_TEST = [
    {"nombre": "hba1c",          "costo_clp": 8000,  "prioridad_clinica": 1,
     "mes_default": 1, "obligatorio_diagnostico": True},
    {"nombre": "glicemia",       "costo_clp": 4000,  "prioridad_clinica": 1,
     "mes_default": 1, "obligatorio_diagnostico": True},
    {"nombre": "perfil_lipidico","costo_clp": 12000, "prioridad_clinica": 2,
     "mes_default": 2, "obligatorio_diagnostico": False},
    {"nombre": "egfr",           "costo_clp": 6000,  "prioridad_clinica": 3,
     "mes_default": 3, "obligatorio_diagnostico": False},
]

def test_capacidad_alta_no_difiere():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=50000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    assert cal.meses_hasta_perfil_completo <= 3

def test_capacidad_baja_difiere_no_obligatorios():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=15000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    examenes_mes_1 = cal.examenes_mes(1)
    assert all(e.obligatorio_diagnostico for e in examenes_mes_1)

def test_obligatorios_siempre_en_mes_1_si_capacidad_minima():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=12000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    obligatorios = cal.examenes_obligatorios_mes_1()
    assert len(obligatorios) == 2

def test_capacidad_insuficiente_total():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=8000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    obligatorios_mes_1 = cal.examenes_obligatorios_mes_1()
    assert len(obligatorios_mes_1) == 1

def test_costo_total_consistente():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=20000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    suma_esperada = sum(e["costo_clp"] for e in EXAMENES_DM2_TEST)
    assert cal.costo_total_clp == suma_esperada

def test_meses_hasta_perfil_completo_capacidad_baja():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=10000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    assert cal.meses_hasta_perfil_completo >= 3

def test_tiene_diferimiento_true_si_mas_1_mes():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=15000,
        examenes_recomendados=EXAMENES_DM2_TEST,
    )
    assert cal.tiene_diferimiento is True

def test_tiene_diferimiento_false_si_solo_1_mes():
    cal = CalendarioExamenes.crear_para_paciente(
        capacidad_pago_clp_mes=100000,
        examenes_recomendados=[{
            "nombre": "hba1c", "costo_clp": 8000, "prioridad_clinica": 1,
            "mes_default": 1,  "obligatorio_diagnostico": True,
        }],
    )
    assert cal.tiene_diferimiento is False

def test_ges_todos_examenes_en_visita_unica():
    cal = CalendarioExamenes.crear_para_paciente_ges(
        examenes_recomendados=EXAMENES_DIAGNOSTICOS_GES_DM2,
    )
    assert len(cal.examenes) == 6
    assert all(ex.mes_programado == 1 for ex in cal.examenes)
    assert cal.meses_hasta_perfil_completo == 1
    assert cal.tiene_diferimiento is False
