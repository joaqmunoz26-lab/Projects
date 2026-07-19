import numpy as np

MICROALBUMINURIA_BASAL_MEANLOG = 2.89
MICROALBUMINURIA_BASAL_SIGMALOG = 0.9
ECG_ANORMAL_BASAL_P = 0.25

DISTRIBUCIONES = {

    "edad": {
        "tipo": "truncnorm",
        "media": 60.0,
        "sd": 10.0,
        "min": 40,
        "max": 80,
    },

    "sexo": {
        "tipo": "categorica",
        "categorias": ["M", "F"],
        "probabilidades": [0.47, 0.53],
    },

    "anios_diagnostico": {
        "tipo": "exponencial",
        "escala": 8.0,
        "min": 0.0,
        "max": 40.0,
    },

    "talla_cm": {
        "tipo": "normal_condicional_sexo",
        "M": {"media": 171, "sd": 7},
        "F": {"media": 158, "sd": 6},
        "min": 140,
        "max": 210,
    },

    "imc_basal": {
        "tipo": "truncnorm",
        "media": 30.5,
        "sd": 5.2,
        "min": 18,
        "max": 50,
    },

    "hba1c_basal": {
        "tipo": "truncnorm",
        "media": 7.4,
        "sd": 1.3,
        "min": 6.5,
        "max": 14.0,
    },

    "glucosa_ayunas_basal": {
        "tipo": "correlacionada_hba1c",
        "slope": 28.7,
        "intercept": -46.7,
        "sd_residual": 20.0,
        "min": 60,
        "max": 400,
    },

    "presion_sistolica_basal": {
        "tipo": "truncnorm",
        "media": 135,
        "sd": 15,
        "min": 90,
        "max": 200,
    },

    "presion_diastolica_basal": {
        "tipo": "truncnorm",
        "media": 82,
        "sd": 10,
        "min": 50,
        "max": 120,
    },

    "colesterol_ldl_basal": {
        "tipo": "truncnorm",
        "media": 110,
        "sd": 32,
        "min": 40,
        "max": 250,
    },

    "funcion_renal_egfr_basal": {
        "tipo": "normal_condicional_edad",
        "formula": "lambda e: 120 - 0.8*(e-40)",
        "sd": 15,
        "min": 15,
        "max": 120,
    },
    "adherencia_basal": {
        "tipo": "beta",
        "alpha": 6.0,
        "beta": 2.0,
    },

    "num_medicamentos": {
        "tipo": "poisson",
        "lambda": 4.5,
        "min": 1,
        "max": 12,
    },

    "hospitalizacion_previa_12m": {
        "tipo": "bernoulli",
        "p": 0.08,
    },

    "microalbuminuria_basal": {
        "tipo": "lognormal",
        "meanlog": MICROALBUMINURIA_BASAL_MEANLOG,
        "sigmalog": MICROALBUMINURIA_BASAL_SIGMALOG,
        "min": 1.0,
        "max": 3000.0,
    },

    "ecg_anormal_basal": {
        "tipo": "bernoulli_condicional",
        "p_base": ECG_ANORMAL_BASAL_P,
    },

}

VARIABILIDAD_TEMPORAL = {
    "hba1c": {"sd_trimestral": 0.35, "sd_medicion": 0.15},
    "glucosa_ayunas": {"sd_trimestral": 15.0, "sd_medicion": 10.0},
    "presion_sistolica": {"sd_trimestral": 8.0, "sd_medicion": 5.0},
    "presion_diastolica": {"sd_trimestral": 5.0, "sd_medicion": 3.0},
    "peso_kg": {"sd_trimestral": 1.5, "sd_medicion": 0.3},
    "colesterol_ldl": {"sd_trimestral": 12.0, "sd_medicion": 6.0},
    "funcion_renal_egfr": {"sd_trimestral": 4.0, "sd_medicion": 2.0},
    "adherencia_tratamiento": {"sd_trimestral": 0.08, "sd_medicion": 0.02},
    "num_medicamentos": {"sd_trimestral": 0.5, "sd_medicion": 0.0},
    "microalbuminuria": {"sd_trimestral": 0.10, "sd_medicion": 0.05},
}


if __name__ == "__main__":
    print("=" * 60)
    print("Distribuciones parametrizadas")
    print("=" * 60)

    print(f"\nTotal variables con distribucion: {len(DISTRIBUCIONES)}")
    print(f"Variables con variabilidad temporal: {len(VARIABILIDAD_TEMPORAL)}")

    print("\nResumen de parametros:")
    for nombre, params in DISTRIBUCIONES.items():
        tipo = params.get("tipo", "?")
        print(f"  - {nombre} ({tipo})")

    print("\nValidando con muestras aleatorias:")
    rng = np.random.default_rng(seed=42)

    edades = []
    for _ in range(1000):
        e = rng.normal(DISTRIBUCIONES["edad"]["media"],
                       DISTRIBUCIONES["edad"]["sd"])
        e = np.clip(e, DISTRIBUCIONES["edad"]["min"],
                    DISTRIBUCIONES["edad"]["max"])
        edades.append(e)

    print(f"  edad: media={np.mean(edades):.1f} sd={np.std(edades):.1f}")
    print("  (esperado: media=60.0 sd=10.0, truncado a [40, 80])")

    print("\nDistribuciones cargadas correctamente.")
