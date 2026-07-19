import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.evolucion_clinica import ModeloEvolucionClinica


def test_modelo_inicializa():
    modelo = ModeloEvolucionClinica(seed=42)
    assert modelo.cambios_registrados == []

def test_via_verde_estable():
    modelo = ModeloEvolucionClinica(seed=42)
    datos  = {"hba1c": 6.5, "adherencia_tratamiento": 0.95}

    deltas = []
    for _ in range(100):
        d = modelo.evolucionar_paciente(datos, "VERDE", 2)
        deltas.append(d["hba1c"] - 6.5)

    assert abs(sum(deltas) / len(deltas)) < 0.1

def test_via_amarilla_mejora_promedio():
    modelo = ModeloEvolucionClinica(seed=42)
    datos  = {"hba1c": 8.0, "adherencia_tratamiento": 0.6}

    deltas = []
    for _ in range(200):
        d = modelo.evolucionar_paciente(datos, "AMARILLA", 2)
        deltas.append(d["hba1c"] - 8.0)

    assert sum(deltas) / len(deltas) < -0.1

def test_via_roja_mejora_mas_que_amarilla():
    m_a = ModeloEvolucionClinica(seed=42)
    m_r = ModeloEvolucionClinica(seed=42)
    datos = {"hba1c": 9.0, "adherencia_tratamiento": 0.5}

    deltas_a, deltas_r = [], []
    for _ in range(200):
        d_a = m_a.evolucionar_paciente(datos, "AMARILLA", 2)
        d_r = m_r.evolucionar_paciente(datos, "ROJA",     2)
        deltas_a.append(d_a["hba1c"] - 9.0)
        deltas_r.append(d_r["hba1c"] - 9.0)

    assert sum(deltas_r) / len(deltas_r) < sum(deltas_a) / len(deltas_a)

def test_glucosa_correlacionada_con_hba1c():
    modelo = ModeloEvolucionClinica(seed=42)
    d = modelo.evolucionar_paciente(
        {"hba1c": 8.0, "adherencia_tratamiento": 0.7},
        "AMARILLA", 2,
    )
    glucosa_esperada = 28.7 * d["hba1c"] - 46.7
    assert abs(d["glucosa_ayunas"] - glucosa_esperada) < 60

def test_seed_reproducibilidad():
    datos = {"hba1c": 7.5, "adherencia_tratamiento": 0.8}
    d1 = ModeloEvolucionClinica(seed=42).evolucionar_paciente(datos, "AMARILLA", 2)
    d2 = ModeloEvolucionClinica(seed=42).evolucionar_paciente(datos, "AMARILLA", 2)
    assert d1["hba1c"] == d2["hba1c"]

def test_estadisticas_evolucion():
    modelo = ModeloEvolucionClinica(seed=42)
    datos  = {"hba1c": 8.0, "adherencia_tratamiento": 0.7}

    for _ in range(50):
        modelo.evolucionar_paciente(datos, "AMARILLA", 2)

    stats = modelo.estadisticas_evolucion()
    assert stats["total_cambios_registrados"] == 50
    assert "tasa_mejora_pct" in stats
    assert stats["mejoras"] + stats["deterioros"] + stats["estables"] == 50

def test_egfr_se_deteriora_con_hba1c_alta():
    m_alto = ModeloEvolucionClinica(seed=42)
    m_bajo = ModeloEvolucionClinica(seed=42)

    datos_a = {"hba1c": 10.0, "funcion_renal_egfr": 80, "adherencia_tratamiento": 0.6}
    datos_b = {"hba1c":  6.5, "funcion_renal_egfr": 80, "adherencia_tratamiento": 0.9}

    deltas_a, deltas_b = [], []
    for _ in range(100):
        d_a = m_alto.evolucionar_paciente(datos_a, "AMARILLA", 2)
        d_b = m_bajo.evolucionar_paciente(datos_b, "VERDE",    2)
        deltas_a.append(d_a["funcion_renal_egfr"] - 80)
        deltas_b.append(d_b["funcion_renal_egfr"] - 80)

    assert sum(deltas_a) / len(deltas_a) < sum(deltas_b) / len(deltas_b)
