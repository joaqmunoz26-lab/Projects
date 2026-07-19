import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.perfiles import ZonaGeografica
from servicios.red_especialistas import (
    CAPACIDAD_DIARIA_POR_ESPECIALISTA,
    RedEspecialistas,
)

U  = ZonaGeografica.URBANO
RC = ZonaGeografica.RURAL_CERCANO
RA = ZonaGeografica.RURAL_AISLADO

@pytest.fixture
def red():
    return RedEspecialistas()

def test_red_carga_10_especialistas(red):
    assert len(red.listar_todos()) == 10

def test_red_todos_son_diabetologos(red):
    assert all(e.especialidad == "diabetologia" for e in red.listar_todos())

def test_red_distribucion_6_urbanos(red):
    urbanos = [e for e in red.listar_todos() if U in e.zonas_cobertura]
    assert len(urbanos) == 6

def test_red_distribucion_3_rural_cercano_exclusivos(red):
    solo_rc = [e for e in red.listar_todos()
               if e.zonas_cobertura == {RC}]
    assert len(solo_rc) == 3

def test_red_distribucion_1_rural_aislado(red):
    ra_especialistas = [e for e in red.listar_todos() if RA in e.zonas_cobertura]
    assert len(ra_especialistas) == 1
    assert ra_especialistas[0].id == "doc_010"

def test_red_ids_secuenciales(red):
    ids = {e.id for e in red.listar_todos()}
    esperados = {f"doc_{i:03d}" for i in range(1, 11)}
    assert ids == esperados

def test_red_capacidad_diaria_1_para_todos(red):
    assert all(e.capacidad_diaria == CAPACIDAD_DIARIA_POR_ESPECIALISTA
               for e in red.listar_todos())

def test_todos_inician_disponibles_y_sin_carga(red):
    for e in red.listar_todos():
        assert e.disponible is True
        assert e.carga_actual_consultas == 0
        assert e.carga_pct == 0.0

def test_carga_pct_100_cuando_asignado(red):
    e = red.buscar_por_id("doc_001")
    e.carga_actual_consultas = 1
    assert e.carga_pct == pytest.approx(100.0)

def test_puede_aceptar_caso_libre(red):
    e = red.buscar_por_id("doc_001")
    assert e.puede_aceptar_caso is True

def test_puede_aceptar_caso_saturado(red):
    e = red.buscar_por_id("doc_001")
    e.carga_actual_consultas = 1
    assert e.puede_aceptar_caso is False

def test_puede_aceptar_caso_no_disponible(red):
    e = red.buscar_por_id("doc_007")
    e.disponible = False
    assert e.puede_aceptar_caso is False

def test_buscar_respaldo_diabetologia_urbano(red):
    candidato = red.buscar_respaldo_mas_cercano("diabetologia", U, excluir_ids={"doc_001"})
    assert candidato is not None
    assert U in candidato.zonas_cobertura

def test_buscar_respaldo_rural_cercano_solo_nativos(red):
    candidato = red.buscar_respaldo_mas_cercano("diabetologia", RC)
    assert candidato is not None
    assert candidato.zonas_cobertura == {RC}

def test_buscar_respaldo_urbano_no_recibe_de_rural_cercano(red):
    for doc_id in [f"doc_{i:03d}" for i in range(1, 7)]:
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    candidato = red.buscar_respaldo_mas_cercano("diabetologia", U)
    assert candidato is None

def test_buscar_respaldo_rural_aislado_solo_doc_010(red):
    candidato = red.buscar_respaldo_mas_cercano("diabetologia", RA)
    assert candidato is not None
    assert candidato.id == "doc_010"

def test_buscar_respaldo_rural_aislado_sin_titular_no_hay_respaldo(red):
    candidato = red.buscar_respaldo_mas_cercano(
        "diabetologia", RA, excluir_ids={"doc_010"})
    assert candidato is None


def test_buscar_respaldo_retorna_menor_carga(red):
    candidato = red.buscar_respaldo_mas_cercano(
        "diabetologia", U,
        excluir_ids={"doc_003", "doc_004", "doc_005", "doc_006"},
    )
    assert candidato is not None
    assert candidato.id in {"doc_001", "doc_002"}

def test_buscar_respaldo_excluye_ids(red):
    candidato = red.buscar_respaldo_mas_cercano(
        "diabetologia", RA, excluir_ids={"doc_010"})
    assert candidato is None


def test_buscar_respaldo_respeta_carga_maxima(red):
    red.buscar_por_id("doc_010").carga_actual_consultas = 1
    candidato = red.buscar_respaldo_mas_cercano(
        "diabetologia", RA, carga_maxima_pct=80.0)
    assert candidato is None

def test_cascade_nativo_urbano(red):
    tipo, eid = red.asignar_con_cascade(U)
    assert tipo == "nativo"
    assert eid in {f"doc_{i:03d}" for i in range(1, 7)}

def test_cascade_nativo_rural_cercano(red):
    tipo, eid = red.asignar_con_cascade(RC)
    assert tipo == "nativo"
    assert eid in {"doc_007", "doc_008", "doc_009"}


def test_cascade_nativo_rural_aislado(red):
    tipo, eid = red.asignar_con_cascade(RA)
    assert tipo == "nativo"
    assert eid == "doc_010"


def test_cascade_prestamo_rural_cercano_cuando_nativos_saturados(red):
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo, eid = red.asignar_con_cascade(RC)
    assert tipo == "prestamo"
    assert eid in {f"doc_{i:03d}" for i in range(1, 7)}

def test_cascade_prestamo_solo_1_por_dia(red):
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo1, _ = red.asignar_con_cascade(RC)
    assert tipo1 == "prestamo"
    tipo2, eid2 = red.asignar_con_cascade(RC)
    assert tipo2 == "saturado"
    assert eid2 is None

def test_cascade_saturado_rural_aislado(red):
    red.buscar_por_id("doc_010").carga_actual_consultas = 1
    tipo, eid = red.asignar_con_cascade(RA)
    assert tipo == "saturado"
    assert eid is None

def test_cascade_saturado_rural_cercano_sin_urbano(red):
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    for doc_id in [f"doc_{i:03d}" for i in range(1, 7)]:
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo, eid = red.asignar_con_cascade(RC)
    assert tipo == "saturado"
    assert eid is None

def test_cascade_incrementa_carga(red):
    tipo, eid = red.asignar_con_cascade(RA)
    assert tipo == "nativo"
    e = red.buscar_por_id(eid)
    assert e.carga_actual_consultas == 1


def test_resetear_cargas_diarias_borra_carga_y_prestamos(red):
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo1, _ = red.asignar_con_cascade(RC)
    assert tipo1 == "prestamo"
    red.buscar_por_id("doc_001").disponible = False

    red.resetear_cargas_diarias()

    assert all(e.carga_actual_consultas == 0 for e in red.listar_todos())
    assert all(e.disponible for e in red.listar_todos())
    tipo2, eid2 = red.asignar_con_cascade(RC)
    assert tipo2 == "nativo"
    assert eid2 in {"doc_007", "doc_008", "doc_009"}


def test_resetear_prestamos_diarios(red):
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo1, _ = red.asignar_con_cascade(RC)
    assert tipo1 == "prestamo"
    tipo2, _ = red.asignar_con_cascade(RC)
    assert tipo2 == "saturado"
    red.resetear_prestamos_diarios()
    for e in red.listar_todos():
        e.carga_actual_consultas = 0
    for doc_id in ("doc_007", "doc_008", "doc_009"):
        red.buscar_por_id(doc_id).carga_actual_consultas = 1
    tipo3, eid3 = red.asignar_con_cascade(RC)
    assert tipo3 == "prestamo"

def test_asignar_caso_incrementa_carga(red):
    red.asignar_caso("doc_010")
    e = red.buscar_por_id("doc_010")
    assert e.carga_actual_consultas == 1

def test_asignar_caso_no_disponible_retorna_false(red):
    e = red.buscar_por_id("doc_007")
    e.disponible = False
    resultado = red.asignar_caso("doc_007")
    assert resultado is False
    assert e.carga_actual_consultas == 0

def test_liberar_caso_decrementa_carga(red):
    red.asignar_caso("doc_001")
    red.liberar_caso("doc_001")
    e = red.buscar_por_id("doc_001")
    assert e.carga_actual_consultas == 0

def test_liberar_caso_no_baja_de_cero(red):
    red.liberar_caso("doc_001")
    e = red.buscar_por_id("doc_001")
    assert e.carga_actual_consultas == 0

def test_estadisticas_totales(red):
    stats = red.estadisticas_red()
    assert stats["total"] == 10
    assert stats["disponibles"] == 10
    assert stats["saturados"] == 0
    assert stats["carga_promedio_pct"] == 0.0

def test_estadisticas_cobertura_rural_aislado(red):
    stats = red.estadisticas_red()
    assert stats["cobertura_por_zona"]["rural_aislado"] == 1

def test_estadisticas_cobertura_urbano(red):
    stats = red.estadisticas_red()
    assert stats["cobertura_por_zona"]["urbano"] == 6


def test_estadisticas_cobertura_rural_cercano(red):
    stats = red.estadisticas_red()
    assert stats["cobertura_por_zona"]["rural_cercano"] == 3

def test_estadisticas_especialidad_unica(red):
    stats = red.estadisticas_red()
    assert stats["cobertura_por_especialidad"] == {"diabetologia": 10}

def test_esta_disponible_retorna_true_para_titular_libre(red):
    assert red.esta_disponible("doc_001") is True

def test_esta_disponible_retorna_false_para_inexistente(red):
    assert red.esta_disponible("medico_inexistente_xyz") is False

def test_esta_disponible_retorna_false_para_saturado(red):
    red.asignar_caso("doc_001")
    assert red.esta_disponible("doc_001") is False

def test_obtener_especialidad_retorna_correcta(red):
    assert red.obtener_especialidad("doc_001") == "diabetologia"

def test_obtener_especialidad_retorna_none_para_inexistente(red):
    assert red.obtener_especialidad("xyz") is None
