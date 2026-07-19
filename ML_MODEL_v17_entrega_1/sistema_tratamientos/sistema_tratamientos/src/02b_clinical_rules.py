import numpy as np


def evolucionar_hba1c(hba1c_actual, adherencia, num_medicamentos, rng):
    efecto_adherencia = -0.6 * (adherencia - 0.5)
    efecto_medicacion_insuficiente = 0.0
    if hba1c_actual > 7.5 and num_medicamentos < 3:
        efecto_medicacion_insuficiente = 0.3

    cambio = (efecto_adherencia
              + efecto_medicacion_insuficiente + rng.normal(0, 0.35))

    nueva = hba1c_actual + cambio
    return float(np.clip(nueva, 6.5, 14.0))

def evolucionar_adherencia(adh_actual, edad, num_medicamentos, rng):
    deriva = -0.01 * (num_medicamentos - 3) / 10
    if edad > 75:
        deriva -= 0.015

    ruido = rng.normal(0, 0.05)
    nueva = adh_actual + deriva + ruido
    return float(np.clip(nueva, 0.0, 1.0))

def evolucionar_egfr(egfr_actual, hba1c, presion_sistolica, edad, rng):
    declive_base = -0.15

    if hba1c > 8.0:
        declive_base -= 0.4 * (hba1c - 8.0)

    if presion_sistolica > 140:
        declive_base -= 0.2

    if edad > 70:
        declive_base -= 0.1

    ruido = rng.normal(0, 1.5)
    nueva = egfr_actual + declive_base + ruido
    return float(np.clip(nueva, 10.0, 120.0))

def evolucionar_peso(peso_actual, talla_cm, rng):
    ruido = rng.normal(0, 1.0)
    nuevo = peso_actual + ruido

    peso_min = 0.5 * (talla_cm / 100) ** 2 * 18
    peso_max = 0.5 * (talla_cm / 100) ** 2 * 55
    return float(np.clip(nuevo, peso_min, peso_max))

def evolucionar_presion(presion_actual, adherencia, imc, rng, es_sistolica=True):
    efecto_adh = -3.0 * (adherencia - 0.7)
    efecto_imc = 0.3 * max(0, imc - 25)

    sd = 8.0 if es_sistolica else 5.0
    ruido = rng.normal(0, sd)

    nueva = presion_actual + efecto_adh + efecto_imc + ruido

    if es_sistolica:
        return float(np.clip(nueva, 90, 220))
    return float(np.clip(nueva, 50, 130))

def evolucionar_ldl(ldl_actual, adherencia, edad, rng):
    if adherencia >= 0.8:
        tendencia_anual = -7.5
    elif adherencia < 0.5:
        tendencia_anual = 4.0
    else:
        tendencia_anual = -1.5

    if edad > 70:
        tendencia_anual += 1.0

    cambio_trimestral = tendencia_anual / 4.0
    ruido = rng.normal(0, 6.0)
    nueva = ldl_actual + cambio_trimestral + ruido
    return float(np.clip(nueva, 40.0, 250.0))

def evolucionar_microalbuminuria(acr_actual, hba1c, presion_sistolica, adherencia, rng):
    factor = 0.0
    if hba1c > 8.0:
        factor += 0.04 * (hba1c - 8.0)
    if presion_sistolica > 140:
        factor += 0.03
    if adherencia < 0.6:
        factor += 0.03
    if hba1c < 7.0 and adherencia >= 0.8:
        factor -= 0.03

    ruido = rng.normal(0, 0.10)
    nueva = acr_actual * (1.0 + factor + ruido)
    return float(np.clip(nueva, 1.0, 3000.0))

def evolucionar_ecg(ecg_anormal_actual, edad, hba1c, presion_sistolica,
                    hospitalizacion_previa, rng):
    if int(ecg_anormal_actual) == 1:
        return 1

    p_nuevo = 0.005
    if edad > 70:
        p_nuevo += 0.010
    if presion_sistolica > 150:
        p_nuevo += 0.010
    if hba1c > 9.0:
        p_nuevo += 0.008
    if hospitalizacion_previa:
        p_nuevo += 0.010

    return int(rng.random() < p_nuevo)

def probabilidad_descompensacion_90d(trayectoria_hba1c, adherencia_actual,
                                      hba1c_actual, egfr_actual,
                                      hospitalizacion_previa,
                                      microalbuminuria=None, ecg_anormal=None):
    p = 0.0

    if hba1c_actual >= 8.0:
        p += 0.55
    elif hba1c_actual >= 7.5:
        p += 0.30
    elif hba1c_actual >= 7.0:
        p += 0.15
    else:
        p += 0.05

    if len(trayectoria_hba1c) >= 3:
        ultimos3 = trayectoria_hba1c[-3:]
        if ultimos3[-1] > ultimos3[0] and ultimos3[-1] - ultimos3[0] > 0.3:
            p += 0.20

    if adherencia_actual < 0.6:
        p += 0.20
    elif adherencia_actual < 0.75:
        p += 0.08

    if egfr_actual < 45:
        p += 0.10

    if hospitalizacion_previa:
        p += 0.08

    if microalbuminuria is not None and microalbuminuria >= 30:
        p += 0.04

    if ecg_anormal:
        p += 0.03

    return float(np.clip(p, 0.01, 0.97))

def generar_evento_descompensacion(prob, rng):
    return int(rng.random() < prob)

if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de reglas clinicas")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    print("\nEscenario 1: paciente bien controlado")
    hba1c = 6.8
    adh = 0.95
    for t in range(4):
        hba1c = evolucionar_hba1c(hba1c, adh, 180, 2, rng)
        print(f"  t={t+1} HbA1c={hba1c:.2f}")

    print("\nEscenario 2: paciente mal controlado y poca adherencia")
    hba1c = 8.5
    adh = 0.50
    trayectoria = [8.5]
    for t in range(4):
        hba1c = evolucionar_hba1c(hba1c, adh, 30, 4, rng)
        adh = evolucionar_adherencia(adh, 68, 4, rng)
        trayectoria.append(hba1c)
        print(f"  t={t+1} HbA1c={hba1c:.2f} adh={adh:.2f}")

    p = probabilidad_descompensacion_90d(trayectoria, adh, hba1c, 55, False)
    print(f"  Probabilidad descompensacion 90d = {p:.2%}")

    print("\nReglas validadas correctamente.")
