import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.fases import FaseActual
from core.paciente_context import PacienteContext
from core.perfiles import (
    CanalDiagnostico,
    CanalMonitoreo,
    PerfilPaciente,
    ZonaGeografica,
)
from motor.reglas_duras import MotorReglasDuras

_OK   = "OK"
_FAIL = "FALLO"
_resultados: list[str] = []

def _assert(condicion: bool, nombre: str) -> None:
    estado = _OK if condicion else _FAIL
    _resultados.append(f"  [{estado}] {nombre}")
    if not condicion:
        raise AssertionError(nombre)

def _get_regla(motor: MotorReglasDuras, regla_id: str):
    return next((r for r in motor.reglas if r.id == regla_id), None)

def _ctx(datos: dict) -> PacienteContext:
    return PacienteContext(
        paciente_id="TEST",
        fase_actual=FaseActual.FASE_2_EWS,
        perfil_paciente=PerfilPaciente(
            zona=ZonaGeografica.URBANO,
            canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
            canal_monitoreo=CanalMonitoreo.APP_AUTONOMA,
        ),
        datos_clinicos=datos,
    )

def test_rd001_glucosa_critica_umbral_300() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD001").validador

    _assert(fn({"glucosa_ayunas": 301}),      "RD001 activa con 301 mg/dL")
    _assert(fn({"glucosa_ayunas": 350}),      "RD001 activa con 350 mg/dL")
    _assert(not fn({"glucosa_ayunas": 300}),  "RD001 NO activa en limite 300")
    _assert(not fn({"glucosa_ayunas": 180}),  "RD001 NO activa con glucosa normal")
    _assert(not fn({}),                       "RD001 NO activa sin variable")

def test_rd002_hipoglucemia_umbral_54() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD002").validador

    _assert(fn({"glucosa_ayunas": 53}),       "RD002 activa con 53 mg/dL")
    _assert(fn({"glucosa_ayunas": 40}),       "RD002 activa con 40 mg/dL")
    _assert(not fn({"glucosa_ayunas": 54}),   "RD002 NO activa en limite 54")
    _assert(not fn({"glucosa_ayunas": 80}),   "RD002 NO activa con glucosa normal")
    _assert(not fn({"glucosa_ayunas": 0}),    "RD002 NO activa con glucosa=0 (dato faltante)")
    _assert(not fn({}),                       "RD002 NO activa sin variable")

def test_rd003_hba1c_descompensacion_severa_umbral_11() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD003").validador

    _assert(fn({"hba1c": 11.1}),      "RD003 activa con 11.1%")
    _assert(fn({"hba1c": 14.0}),      "RD003 activa con 14.0%")
    _assert(not fn({"hba1c": 11.0}),  "RD003 NO activa en limite 11.0%")
    _assert(not fn({"hba1c": 8.5}),   "RD003 NO activa con HbA1c moderada")
    _assert(not fn({}),               "RD003 NO activa sin variable")

def test_rd005_egfr_nefropatia_avanzada_umbral_30() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD005").validador

    _assert(fn({"funcion_renal_egfr": 29}),      "RD005 activa con eGFR 29")
    _assert(fn({"funcion_renal_egfr": 15}),      "RD005 activa con eGFR 15 (G5)")
    _assert(not fn({"funcion_renal_egfr": 30}),  "RD005 NO activa en limite 30")
    _assert(not fn({"funcion_renal_egfr": 60}),  "RD005 NO activa con eGFR normal")
    _assert(not fn({"funcion_renal_egfr": 0}),   "RD005 NO activa con eGFR=0 (dato faltante)")
    _assert(not fn({}),                          "RD005 NO activa sin variable")

def test_rd004_pas_alta_activa_crisis() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD004").validador

    _assert(fn({"presion_sistolica": 181, "presion_diastolica": 85}),
            "RD004 activa con PAS=181 (sobre umbral 180)")
    _assert(fn({"presion_sistolica": 200, "presion_diastolica": 100}),
            "RD004 activa con PAS=200")
    _assert(fn({"presion_sistolica": 180, "presion_diastolica": 85}),
            "RD004 activa en limite PAS=180")
    _assert(not fn({"presion_sistolica": 140, "presion_diastolica": 90}),
            "RD004 NO activa con HTA grado 1")

def test_rd004_pad_alta_activa_crisis() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD004").validador

    _assert(fn({"presion_sistolica": 160, "presion_diastolica": 111}),
            "RD004 activa con PAD=111 (sobre umbral 110)")
    _assert(fn({"presion_diastolica": 125}),
            "RD004 activa con solo PAD disponible >= 110")
    _assert(fn({"presion_sistolica": 150, "presion_diastolica": 110}),
            "RD004 activa en limite PAD=110 (inclusivo, ESH 2024 Fig. 5)")

def test_rd004_sin_variables_no_activa() -> None:
    motor = MotorReglasDuras()
    fn    = _get_regla(motor, "RD004").validador

    _assert(not fn({}),                          "RD004 NO activa sin variables de PA")
    _assert(not fn({"glucosa_ayunas": 150}),     "RD004 NO activa con solo glucosa")

def test_rd004_codigo_y_prioridad_en_motor() -> None:
    motor = MotorReglasDuras()
    rd004 = _get_regla(motor, "RD004")

    _assert(rd004 is not None,             "RD004 existe en el motor")
    _assert(rd004.codigo == "RD004",       "codigo == 'RD004' (alias de id)")
    _assert(rd004.severidad == "CRITICA",  "severidad == 'CRITICA'")

def test_rd004_motor_reglas_criticas_incluye_rd004() -> None:
    motor = MotorReglasDuras()
    codigos_criticos = [r.codigo for r in motor.reglas_criticas]

    _assert("RD004" in codigos_criticos,     "RD004 esta en reglas_criticas del motor")
    _assert("RD001" in codigos_criticos,     "RD001 esta en reglas_criticas del motor")
    _assert("RD002" in codigos_criticos,     "RD002 esta en reglas_criticas del motor")
    _assert("RD003" not in codigos_criticos, "RD003 NO esta en reglas_criticas (es ALTA)")


def test_rd004_bypass_ml_en_pipeline() -> None:
    motor    = MotorReglasDuras()
    ctx      = _ctx({"presion_sistolica": 190, "presion_diastolica": 85})
    resultado = motor.evaluar(ctx)

    _assert(resultado.matcheo,                    "RD004 matcheo=True")
    _assert(resultado.regla_activada == "RD004",  "regla_activada == 'RD004'")
    _assert(resultado.prioridad == "CRITICA",      "prioridad == 'CRITICA' (alias)")
    _assert(resultado.severidad == "CRITICA",      "severidad == 'CRITICA' (campo base)")

if __name__ == "__main__":
    tests = [
        test_rd001_glucosa_critica_umbral_300,
        test_rd002_hipoglucemia_umbral_54,
        test_rd003_hba1c_descompensacion_severa_umbral_11,
        test_rd005_egfr_nefropatia_avanzada_umbral_30,
        test_rd004_pas_alta_activa_crisis,
        test_rd004_pad_alta_activa_crisis,
        test_rd004_sin_variables_no_activa,
        test_rd004_codigo_y_prioridad_en_motor,
        test_rd004_motor_reglas_criticas_incluye_rd004,
        test_rd004_bypass_ml_en_pipeline,
    ]

    errores: list[str] = []
    for fn in tests:
        nombre = fn.__name__
        try:
            fn()
            print(f"  [OK] {nombre}")
        except AssertionError as e:
            print(f"  [FALLO] {nombre}: {e}")
            errores.append(nombre)
        except Exception as e:
            print(f"  [ERROR] {nombre}: {e}")
            errores.append(nombre)

    print()
    if errores:
        print(f"RESULTADO: {len(errores)} test(s) fallaron -> {errores}")
        sys.exit(1)
    else:
        print(f"RESULTADO: {len(tests)}/{len(tests)} tests pasaron")
