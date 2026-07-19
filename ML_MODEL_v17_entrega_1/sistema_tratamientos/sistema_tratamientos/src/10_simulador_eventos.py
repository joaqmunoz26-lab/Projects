import argparse
import logging
import random as _random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from core.decisiones import BarreraActivada, ViaRuteo
from core.fases import FaseActual
from core.perfiles import CanalDiagnostico
from core.perfiles_tipicos import (
    perfil_rural_digital,
    perfil_rural_posta,
    perfil_urbano_digital,
)
from pipeline.fase1_ingesta import asignar_canal_monitoreo
from pipeline.fase3_fallback import set_red_especialistas
from pipeline.orquestador_principal import procesar_paciente
from servicios.hitl_revision import ColaRevisionHITL, MotorEscalamientoHITL
from servicios.red_especialistas import RedEspecialistas

FEATURES_MODELO = import_module("04_features").FEATURES_MODELO

RAIZ        = Path(__file__).resolve().parent.parent
RUTA_YAML   = RAIZ / "00_parametros_logisticos.yaml"
DIR_REPORTS = RAIZ / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger(__name__)

COSTOS_EVENTO_EWS = {
    "via_verde_asincrona": {
        "costo_consulta_clp":      2_158,
        "incluye_viaje":           False,
        "tiempo_paciente_min":     0,
        "tiempo_especialista_min": 8,
    },
    "via_amarilla_telemedicina": {
        "costo_consulta_clp":      8_630,
        "incluye_viaje":           False,
        "tiempo_paciente_min":     20,
        "tiempo_especialista_min": 25,
    },
    "via_amarilla_presencial": {
        "costo_consulta_clp":      10_380,
        "incluye_viaje":           True,
        "tiempo_paciente_min":     140,
        "tiempo_especialista_min": 35,
    },
    "via_roja_presencial": {
        "costo_consulta_clp":      10_380,
        "incluye_viaje":           True,
        "tiempo_paciente_min":     140,
        "tiempo_especialista_min": 45,
    },
}

def _costo_viaje_zona(zona_value: str) -> int:
    zona_lower = zona_value.lower()
    if "aislado" in zona_lower:
        return 30_000
    if "rural" in zona_lower:
        return 8_000
    return 2_000

TIEMPOS_IDA_MIN: dict[str, int] = {
    "urbano":        30,
    "rural_cercano": 60,
    "rural_aislado": 120,
}

def _tiempo_paciente_presencial_min(zona_val: str) -> int:
    ida = TIEMPOS_IDA_MIN.get(zona_val, 60)
    return 2 * ida + 20

CONTROLES_POR_ANO_PROMEDIO = 4.0

PROB_PROFESIONAL_NO_DISPONIBLE_BASE = 0.10
COSTO_REPROGRAMACION_ADMIN_CLP = 500

TASA_NO_SHOW_PACIENTE_BASE = 0.12
COSTO_CUPO_PERDIDO_CLP = 10_380

P_INASISTENCIA_VIA_EWS = {
    "VERDE":    0.00,
    "AMARILLA": 0.10,
    "ROJA":     0.05,
}

DISTRIBUCION_CAPACIDAD_MOVILIZACION = {
    "urbano":        (35000, 60000),
    "rural_cercano": (15000, 30000),
    "rural_aislado": (5000,  18000),
}

DISTRIBUCION_TRAMO_FONASA = {
    "urbano":        {"A": 0.20, "B": 0.30, "C": 0.30, "D": 0.20},
    "rural_cercano": {"A": 0.50, "B": 0.30, "C": 0.15, "D": 0.05},
    "rural_aislado": {"A": 0.70, "B": 0.25, "C": 0.05, "D": 0.00},
}

PROB_INTERNET_DOMICILIO = {
    "urbano":        0.95,
    "rural_cercano": 0.75,
    "rural_aislado": 0.40,
}

def _costo_saturacion_pool(via: str, perfil) -> int:
    from core.perfiles import CanalMonitoreo
    es_presencial = (
        via == "ROJA" or
        (via == "AMARILLA" and
         perfil.canal_monitoreo not in (
             CanalMonitoreo.APP_AUTONOMA,
             CanalMonitoreo.WHATSAPP_AUTONOMO,
         ))
    )
    viaje = _costo_viaje_zona(perfil.zona.value) if es_presencial else 0
    return 2 * viaje + COSTO_REPROGRAMACION_ADMIN_CLP

def cargar_parametros(fuente: str = "asumido_literatura") -> dict:
    with open(RUTA_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    fuentes = data.get("fuentes", {})
    if fuente not in fuentes:
        raise ValueError(f"Fuente '{fuente}' no encontrada. Disponibles: {list(fuentes)}")
    p = fuentes[fuente]
    for sec in ("tiempos", "costos_clp", "capacidad", "flujo_pacientes"):
        if sec not in p:
            raise ValueError(f"Sección '{sec}' faltante en fuente '{fuente}'")
    return p

def cargar_parametros_logisticos(fuente: str = "asumido_literatura") -> dict:
    return cargar_parametros(fuente)

def _samplear(spec: dict, rng) -> float:
    dist = spec.get("distribucion")
    if dist == "normal":
        return max(float(spec.get("minimo", 0)), rng.normal(spec["media"], spec.get("sd", 0)))
    if dist == "exponencial":
        return rng.exponential(spec["escala"])
    return float(spec.get("media", 0))

def _generar_pacientes(n: int, params: dict, rng) -> list:
    _PROP_URBANO         = 0.85
    _PROP_RURAL_CERCANO  = 0.10

    pacientes = []
    for i in range(n):
        hba1c  = float(np.clip(rng.normal(7.0, 1.5), 5.0, 14.0))
        adh    = float(np.clip(rng.beta(4, 2), 0.0, 1.0))
        riesgo = float(np.clip(rng.beta(2, 5), 0.05, 0.95))

        r_zona = rng.random()
        if r_zona < _PROP_URBANO:
            zona = "URBANO"
        elif r_zona < _PROP_URBANO + _PROP_RURAL_CERCANO:
            zona = "RURAL_CERCANO"
        else:
            zona = "RURAL_AISLADO"

        pacientes.append({
            "id":       f"P{i:05d}",
            "zona":     zona,
            "es_rural": zona != "URBANO",
            "riesgo":   riesgo,
            "datos": {
                "hba1c":                    round(hba1c, 2),
                "adherencia_tratamiento":   round(adh, 3),
                "edad":                     int(rng.integers(40, 80)),
                "datos_remotos_disponibles": bool(rng.random() < 0.6),
                "sintomas_activos":          bool(rng.random() < 0.1),
            },
            "fase": "tratamiento",
            "stats": {k: 0 for k in (
                "num_eventos", "num_presenciales", "num_telemedicina",
                "num_asincrona", "num_sin_consulta",
                "tiempo_total_min", "tiempo_viaje_min", "tiempo_espera_min",
                "costo_total_clp", "costo_consulta_clp", "costo_viaje_clp",
                "rebotes", "no_shows",
            )},
        })
    return pacientes

def _acumular(s: dict, t_viaje, t_espera, t_dur, costo_cons, costo_vj):
    s["tiempo_total_min"]   += t_viaje + t_espera + t_dur
    s["tiempo_viaje_min"]   += t_viaje
    s["tiempo_espera_min"]  += t_espera
    s["costo_consulta_clp"] += costo_cons
    s["costo_viaje_clp"]    += costo_vj
    s["costo_total_clp"]    += costo_cons + costo_vj

def _registrar(log_ev, pac, t_sim, modalidad, t_dur, t_viaje, t_espera,
               costo_total, es_rural, escenario):
    log_ev.append({
        "paciente_id":  pac["id"],
        "tiempo_min":   round(t_sim),
        "modalidad":    modalidad,
        "duracion_min": round(t_dur, 1),
        "t_viaje_min":  round(t_viaje, 1),
        "t_espera_min": round(t_espera, 1),
        "costo_clp":    round(costo_total),
        "es_rural":     es_rural,
        "escenario":    escenario,
    })

def _proceso_paciente(env, pac, recursos, params, rng,
                      log_ev, dur_sim):
    yield env.timeout(rng.uniform(0, 90 * 1440))

    fp = params["flujo_pacientes"]
    tp = params["tiempos"]
    cp = params["costos_clp"]
    intervalo_spec = fp.get("intervalo_control_dias", {"media": 90, "sd": 15})

    zona_pac = pac.get("zona", "RURAL_CERCANO" if pac["es_rural"] else "URBANO")
    _acumular(pac["stats"], 0, 0, 0, 10_380, 0)
    pac["stats"]["num_eventos"] += 1
    _registrar(log_ev, pac, env.now, "PRESENCIAL_INGRESO",
               0, 0, 0, 10_380, pac["es_rural"], "base")
    if pac["es_rural"]:
        _rev_mod, _rev_cons, _rev_vj = "PRESENCIAL_REVISION", 10_380, _costo_viaje_zona(zona_pac)
    else:
        _rev_mod, _rev_cons, _rev_vj = "TELEMEDICINA_REVISION", 8_630, 0
    _acumular(pac["stats"], 0, 0, 0, _rev_cons, _rev_vj)
    pac["stats"]["num_eventos"] += 1
    _registrar(log_ev, pac, env.now, _rev_mod,
               0, 0, 0, _rev_cons + _rev_vj, pac["es_rural"], "base")

    while env.now < dur_sim:
        yield env.timeout(max(30.0, _samplear(intervalo_spec, rng)) * 1440)
        if env.now >= dur_sim:
            break

        modalidad = "PRESENCIAL_CONTROL"

        t_viaje = t_espera = t_dur = costo_cons = costo_vj = 0.0
        es_rural = pac["es_rural"]

        if modalidad in ("PRESENCIAL_URGENTE", "PRESENCIAL_CONTROL"):
            zona_pac = pac.get("zona", "RURAL_CERCANO" if es_rural else "URBANO")
            if zona_pac == "RURAL_AISLADO":
                costo_vj = cp["costo_viaje_paciente_rural_aislado_clp"]["valor"]
            elif zona_pac == "RURAL_CERCANO":
                costo_vj = cp["costo_viaje_paciente_rural_cercano_clp"]["valor"]
            else:
                costo_vj = cp["costo_viaje_paciente_urbano_clp"]["valor"]

            if rng.random() < TASA_NO_SHOW_PACIENTE_BASE:
                pac["stats"]["no_shows"] += 1
                pac["stats"]["num_eventos"] += 1
                _acumular(pac["stats"], 0, 0, 0,
                          COSTO_CUPO_PERDIDO_CLP + COSTO_REPROGRAMACION_ADMIN_CLP, 0)
                _registrar(log_ev, pac, env.now, "PRESENCIAL_NO_SHOW",
                           0, 0, 0,
                           COSTO_CUPO_PERDIDO_CLP + COSTO_REPROGRAMACION_ADMIN_CLP,
                           es_rural, "base")
                yield env.timeout(float(np.clip(rng.normal(14, 3), 7, 21)) * 1440)
                if env.now >= dur_sim:
                    continue

            t_viaje = _samplear(tp["viaje_rural_min" if es_rural else "viaje_urbano_min"], rng)
            yield env.timeout(t_viaje)

            if rng.random() < PROB_PROFESIONAL_NO_DISPONIBLE_BASE:
                pac["stats"]["rebotes"] += 1
                pac["stats"]["num_eventos"] += 1
                _acumular(pac["stats"], t_viaje, 0, 0, COSTO_REPROGRAMACION_ADMIN_CLP, costo_vj)
                _registrar(log_ev, pac, env.now, "PRESENCIAL_PROFESIONAL_NO_DISPONIBLE",
                           0, t_viaje, 0, costo_vj + COSTO_REPROGRAMACION_ADMIN_CLP,
                           es_rural, "base")
                yield env.timeout(float(np.clip(rng.normal(14, 3), 7, 21)) * 1440)
                if env.now >= dur_sim:
                    continue
                t_viaje = _samplear(tp["viaje_rural_min" if es_rural else "viaje_urbano_min"], rng)
                yield env.timeout(t_viaje)

            t0 = env.now
            with recursos["presencial"].request() as req:
                yield req
                t_espera = env.now - t0
                t_dur = _samplear(tp["duracion_consulta_presencial_min"], rng)
                yield env.timeout(t_dur)
            costo_cons = cp["consulta_presencial"]["valor"]
            pac["stats"]["num_presenciales"] += 1

        elif modalidad == "TELEMEDICINA_SINCRONICA":
            t0 = env.now
            with recursos["telemedicina"].request() as req:
                yield req
                t_espera = env.now - t0
                if rng.random() >= fp["tasa_no_show_telemedicina"]["valor"]:
                    t_dur = _samplear(tp["duracion_telemedicina_min"], rng)
                    yield env.timeout(t_dur)
            costo_cons = cp["telemedicina_sincronica"]["valor"]
            pac["stats"]["num_telemedicina"] += 1

        elif modalidad == "REVISION_ASINCRONA":
            with recursos["asincrona"].request() as req:
                yield req
                t_dur = _samplear(tp["duracion_revision_asincrona_min"], rng)
                yield env.timeout(t_dur)
            costo_cons = cp["revision_asincrona"]["valor"]
            pac["stats"]["num_asincrona"] += 1

        else:
            pac["stats"]["num_sin_consulta"] += 1

        pac["stats"]["num_eventos"] += 1
        _acumular(pac["stats"], t_viaje, t_espera, t_dur, costo_cons, costo_vj)
        _registrar(log_ev, pac, env.now, modalidad, t_dur, t_viaje, t_espera,
                   costo_cons + costo_vj, es_rural, "base")

def correr_simulacion(escenario: str, n_pacientes: int, dias: int,
                      params: dict, seed: int):
    rng     = np.random.default_rng(seed)
    env     = simpy.Environment()
    dur_sim = dias * 1440

    cap = params["capacidad"]
    recursos = {
        "presencial":  simpy.Resource(env, capacity=cap["cupos_presenciales_dia"]["valor"]),
        "telemedicina": simpy.Resource(env, capacity=cap["cupos_telemedicina_dia"]["valor"]),
        "asincrona":   simpy.Resource(env, capacity=cap["cupos_revision_asincrona_dia"]["valor"]),
    }

    if escenario != "base":
        raise ValueError(f"Escenario desconocido: '{escenario}'")

    pacientes = _generar_pacientes(n_pacientes, params, rng)
    log_ev    = []

    for p in pacientes:
        env.process(_proceso_paciente(
            env, p, recursos, params, rng, log_ev, dur_sim
        ))

    _log.info(f"{escenario} | {n_pacientes} pacientes | {dias} días")
    env.run(until=dur_sim)
    _log.info(f"{escenario} lista | {len(log_ev):,} eventos")
    return pacientes, log_ev

def generar_reportes(pacientes: list, log_ev: list, escenario: str,
                     seed_sufijo: str = "") -> dict:
    out = RAIZ / "reports"
    out.mkdir(exist_ok=True)

    rows = [{**{"paciente_id": p["id"], "es_rural": p["es_rural"]}, **p["stats"]}
            for p in pacientes]
    df_p = pd.DataFrame(rows)
    df_p.to_csv(out / f"simulacion_{escenario}_pacientes{seed_sufijo}.csv", index=False)
    pd.DataFrame(log_ev).to_csv(out / f"simulacion_{escenario}_eventos{seed_sufijo}.csv", index=False)

    resumen = {
        "escenario":              escenario,
        "n_pacientes":            len(pacientes),
        "total_eventos":          int(df_p["num_eventos"].sum()),
        "total_presenciales":     int(df_p["num_presenciales"].sum()),
        "total_telemedicina":     int(df_p["num_telemedicina"].sum()),
        "total_asincrona":        int(df_p["num_asincrona"].sum()),
        "total_sin_consulta":     int(df_p["num_sin_consulta"].sum()),
        "costo_total_clp":        int(df_p["costo_total_clp"].sum()),
        "tiempo_total_min":       int(df_p["tiempo_total_min"].sum()),
        "tiempo_viaje_total_min": int(df_p["tiempo_viaje_min"].sum()),
        "tiempo_espera_total_min":int(df_p["tiempo_espera_min"].sum()),
        "rebotes_totales":        int(df_p["rebotes"].sum()),
        "no_shows_totales":       int(df_p["no_shows"].sum()),
    }
    pd.DataFrame([resumen]).to_csv(out / f"simulacion_{escenario}_resumen{seed_sufijo}.csv", index=False)
    return resumen

def _datos_clinicos_desde_fila(row) -> dict:
    datos = {}
    for f in FEATURES_MODELO:
        val = row[f]
        if f == "sexo":
            datos[f] = int(str(val).upper() == "M") if isinstance(val, str) else int(val)
        elif f in ("ecg_anormal", "hospitalizacion_previa_12m"):
            datos[f] = int(val)
        else:
            datos[f] = float(val)
    return datos

def construir_lote_pacientes_ews(
    pacientes_disponibles,
    df_pacientes: pd.DataFrame,
    n_pacientes: int,
    params: dict,
    seed: int,
    dias_simulacion: int = 365,
) -> list:
    rng      = np.random.default_rng(seed)
    ahora    = datetime.now()

    n_sample = min(n_pacientes, len(pacientes_disponibles))
    ids_muestra = rng.choice(pacientes_disponibles, size=n_sample, replace=False)

    df_muestra = (df_pacientes[df_pacientes["paciente_id"].isin(ids_muestra)]
                  .drop_duplicates("paciente_id", keep="last"))

    from dataclasses import replace as _dc_replace

    lote = []
    for _, row in df_muestra.iterrows():
        r = rng.random()
        if r < 0.85:
            perfil = perfil_urbano_digital()
        elif r < 0.95:
            perfil = perfil_rural_digital(aislado=False)
        else:
            perfil = perfil_rural_posta()

        zona_str   = perfil.zona.value
        rango      = DISTRIBUCION_CAPACIDAD_MOVILIZACION.get(zona_str, (20000, 30000))
        capacidad  = int(rng.integers(rango[0], rango[1] + 1))

        dist_tramo  = DISTRIBUCION_TRAMO_FONASA.get(zona_str, {"A": 0.25, "B": 0.50, "C": 0.25, "D": 0.00})
        r_tramo     = rng.random()
        acum_tramo  = 0.0
        tramo_elegido = "B"
        for letra, prob in dist_tramo.items():
            acum_tramo += prob
            if r_tramo < acum_tramo:
                tramo_elegido = letra
                break

        prob_inet  = PROB_INTERNET_DOMICILIO.get(zona_str, 0.50)
        tiene_inet = bool(rng.random() < prob_inet)

        perfil = _dc_replace(perfil,
                             canal_diagnostico=CanalDiagnostico.PRESENCIAL_CENTRO,
                             capacidad_movilizacion_clp_mes=capacidad,
                             tramo_fonasa=tramo_elegido,
                             acceso_internet_domicilio=tiene_inet)

        datos_clinicos = _datos_clinicos_desde_fila(row)

        lote.append({
            "_dia_llegada":                  int(rng.integers(0, dias_simulacion)),
            "paciente_id":                   str(row["paciente_id"]),
            "datos_clinicos":                datos_clinicos,
            "perfil":                        perfil,
            "fase_inicial":                  None,
            "dm2_confirmado_inicial":        False,
            "fecha_proximo_control_rutina":  ahora + timedelta(days=90),
            "especialista_titular":          str(rng.choice([f"doc_{i:03d}" for i in range(1, 11)])),
            "evento":                        "control_programado",
        })
    return lote

def filtrar_pacientes_dia(lote_pacientes: list, dia: int, dias_simulacion: int) -> list:
    return [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in lote_pacientes
        if p["_dia_llegada"] == dia
    ]

def actualizar_disponibilidad_red(red: RedEspecialistas, dia: int, params: dict) -> None:
    todos = red.listar_todos()
    if dia % 7 == 6:
        disponibles = {"doc_001", "doc_002", "doc_007", "doc_010"}
        for e in todos:
            e.disponible = e.id in disponibles
    elif 60 <= dia <= 90 or 210 <= dia <= 240:
        disponibles = {
            "doc_001", "doc_002", "doc_003", "doc_004", "doc_005",
            "doc_007", "doc_008", "doc_010",
        }
        for e in todos:
            e.disponible = e.id in disponibles

def liberar_cupos_diarios(red: RedEspecialistas, dia: int, params: dict) -> None:
    red.resetear_cargas_diarias()

def actualizar_metricas_paciente(metricas: dict, ctx) -> None:
    perfil = ctx.perfil_paciente
    tramo  = getattr(perfil, "tramo_fonasa", "B")
    if isinstance(metricas.get("distribucion_tramo_fonasa"), dict):
        metricas["distribucion_tramo_fonasa"][tramo] = (
            metricas["distribucion_tramo_fonasa"].get(tramo, 0) + 1
        )
    capacidad_movilizacion = getattr(perfil, "capacidad_movilizacion_clp_mes", 30000)
    if capacidad_movilizacion < 12_000:
        metricas["pacientes_logisticamente_vulnerables"] += 1

    if not getattr(perfil, "acceso_internet_domicilio", True):
        metricas["pacientes_sin_internet_domicilio"] += 1
    canal = perfil.canal_monitoreo
    from core.perfiles import CanalMonitoreo as _CM
    if canal == _CM.APP_AUTONOMA:
        metricas["pacientes_canal_app_asignado"] += 1
    elif canal == _CM.TENS_INGRESA:
        metricas["pacientes_canal_tens_asignado"] += 1
    elif canal == _CM.WHATSAPP_AUTONOMO:
        metricas["pacientes_canal_whatsapp_asignado"] += 1

    if ctx.dm2_confirmado:
        metricas["fase1_confirmados"] += 1
    else:
        metricas["fase1_no_confirmados"] += 1
        return

    if ctx.via_ruteo == ViaRuteo.VERDE:
        metricas["fase2_via_verde"] += 1
    elif ctx.via_ruteo == ViaRuteo.AMARILLA:
        metricas["fase2_via_amarilla"] += 1
        if ctx.control_rutina_adelantado:
            metricas["control_rutina_adelantado"] += 1
    elif ctx.via_ruteo == ViaRuteo.ROJA:
        metricas["fase2_via_roja"] += 1

    if ctx.barrera_activada == BarreraActivada.REGLAS_DURAS:
        metricas["barrera_duras_bypass"] += 1

    if ctx.especialista_respaldo_usado:
        metricas["uso_fallback_red"] += 1

    if ctx.especialista_asignado is None and ctx.via_ruteo == ViaRuteo.ROJA:
        metricas["fallback_fallo_red_saturada"] += 1

    if ctx.requiere_hitl:
        metricas["hitl_encolado"] += 1

def extraer_evento_paciente(ctx, dia: int) -> dict:
    return {
        "dia_simulacion":             dia,
        "paciente_id":                ctx.paciente_id,
        "dm2_confirmado":             ctx.dm2_confirmado,
        "via_ruteo":                  ctx.via_ruteo.value if ctx.via_ruteo else None,
        "barrera_activada":           ctx.barrera_activada.value,
        "riesgo_ml":                  ctx.riesgo_ml,
        "especialista_asignado":      ctx.especialista_asignado,
        "especialista_respaldo_usado":ctx.especialista_respaldo_usado,
        "requiere_hitl":              ctx.requiere_hitl,
        "control_rutina_adelantado":  ctx.control_rutina_adelantado,
        "n_decisiones_auditadas":     len(ctx.auditoria),
        "zona":                       ctx.perfil_paciente.zona.value,
    }

def guardar_resultados_ews(
    metricas: dict,
    eventos: list,
    decisiones: list,
    cola_hitl: ColaRevisionHITL,
    motor_escalamiento: MotorEscalamientoHITL,
    sufijo: str = "completo",
    seed_sufijo: str = "",
) -> None:
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metricas]).to_csv(
        DIR_REPORTS / f"simulacion_{sufijo}_resumen{seed_sufijo}.csv", index=False
    )
    pd.DataFrame(eventos).to_csv(
        DIR_REPORTS / f"simulacion_{sufijo}_eventos{seed_sufijo}.csv", index=False
    )
    pd.DataFrame(decisiones).to_csv(
        DIR_REPORTS / f"simulacion_{sufijo}_decisiones{seed_sufijo}.csv", index=False
    )

    solicitudes_hitl = [
        {
            "id":                  sol.id,
            "paciente_id":         sol.paciente_id,
            "tipo":                sol.tipo.value,
            "estado":              sol.estado.value,
            "prioridad":           sol.prioridad.value,
            "creada_en":           str(sol.creada_en),
            "horas_transcurridas": sol.horas_transcurridas,
            "n_escalamientos":     len(sol.escalamientos),
        }
        for sol in cola_hitl._solicitudes.values()
    ]
    pd.DataFrame(solicitudes_hitl).to_csv(
        DIR_REPORTS / f"simulacion_{sufijo}_hitl{seed_sufijo}.csv", index=False
    )

    eventos_escal = [
        {
            "solicitud_id":           e.solicitud_id,
            "fase_protocolo":         e.fase_protocolo,
            "ejecutada_en":           str(e.ejecutada_en),
            "resultado":              e.resultado,
            "especialista_notificado":e.especialista_notificado,
            "especialista_reasignado":e.especialista_reasignado,
        }
        for e in motor_escalamiento.eventos
    ]
    pd.DataFrame(eventos_escal).to_csv(
        DIR_REPORTS / f"simulacion_{sufijo}_escalamientos{seed_sufijo}.csv", index=False
    )

def generar_eventos_anuales_por_paciente(ctx, rng) -> list:
    eventos = []
    zona_val = ctx.perfil_paciente.zona.value
    costo_viaje_dx = _costo_viaje_zona(zona_val)

    if not ctx.dm2_confirmado:
        eventos.append({
            "paciente_id":             ctx.paciente_id,
            "tipo_evento":             "diagnostico_fase1",
            "modalidad":               "presencial_diagnostico",
            "control_num":             1,
            "costo_consulta_clp":      10_380,
            "costo_viaje_clp":         costo_viaje_dx,
            "costo_clp":               10_380 + costo_viaje_dx,
            "tiempo_paciente_min":     _tiempo_paciente_presencial_min(zona_val),
            "tiempo_especialista_min": 30,
        })
        return eventos

    via = ctx.via_ruteo.value if ctx.via_ruteo else None

    if via == "verde":
        tipo_evento = "via_verde_asincrona"
    elif via == "amarilla":
        es_tele = (ctx.perfil_paciente.canal_monitoreo.value
                   in ("app_autonoma", "whatsapp_autonomo"))
        tipo_evento = "via_amarilla_telemedicina" if es_tele else "via_amarilla_presencial"
    elif via == "roja":
        tipo_evento = "via_roja_presencial"
    else:
        tipo_evento = "via_amarilla_presencial"

    config = COSTOS_EVENTO_EWS[tipo_evento]
    n_controles = int(round(CONTROLES_POR_ANO_PROMEDIO))
    costo_viaje = _costo_viaje_zona(zona_val) if config["incluye_viaje"] else 0

    for control_num in range(n_controles):
        costo_cons = config["costo_consulta_clp"]
        eventos.append({
            "paciente_id":             ctx.paciente_id,
            "tipo_evento":             tipo_evento,
            "modalidad":               tipo_evento.split("_")[-1],
            "control_num":             control_num + 1,
            "costo_consulta_clp":      costo_cons,
            "costo_viaje_clp":         costo_viaje,
            "costo_clp":               costo_cons + costo_viaje,
            "tiempo_paciente_min":     (
                _tiempo_paciente_presencial_min(zona_val)
                if config["incluye_viaje"] else config["tiempo_paciente_min"]
            ),
            "tiempo_especialista_min": config["tiempo_especialista_min"],
        })

    return eventos

def generar_evento_de_control(paciente_id, control_num, via, perfil, rng):
    from core.perfiles import CanalMonitoreo

    if via == "VERDE":
        config   = COSTOS_EVENTO_EWS["via_verde_asincrona"]
        modalidad = "asincrona"
    elif via == "AMARILLA":
        es_tele = perfil.canal_monitoreo in (
            CanalMonitoreo.APP_AUTONOMA,
            CanalMonitoreo.WHATSAPP_AUTONOMO,
        )
        if es_tele:
            config   = COSTOS_EVENTO_EWS["via_amarilla_telemedicina"]
            modalidad = "telemedicina"
        else:
            config   = COSTOS_EVENTO_EWS["via_amarilla_presencial"]
            modalidad = "presencial_control"
    elif via == "ROJA":
        config   = COSTOS_EVENTO_EWS["via_roja_presencial"]
        modalidad = "presencial_urgente"
    else:
        config   = COSTOS_EVENTO_EWS["via_amarilla_presencial"]
        modalidad = "presencial_control"

    zona_val = perfil.zona.value
    costo_viaje = _costo_viaje_zona(zona_val) if config["incluye_viaje"] else 0
    costo_cons  = config["costo_consulta_clp"]
    return {
        "paciente_id":             paciente_id,
        "control_num":             control_num,
        "tipo_evento":             f"control_via_{via.lower()}",
        "modalidad":               modalidad,
        "via_clinica":             via,
        "costo_consulta_clp":      costo_cons,
        "costo_viaje_clp":         costo_viaje,
        "costo_clp":               costo_cons + costo_viaje,
        "tiempo_paciente_min":     (
            _tiempo_paciente_presencial_min(zona_val)
            if config["incluye_viaje"] else config["tiempo_paciente_min"]
        ),
        "tiempo_especialista_min": config["tiempo_especialista_min"],
    }

def _evento_revision_fase1(ctx) -> dict:
    from core.perfiles import CanalMonitoreo
    canal    = asignar_canal_monitoreo(ctx.perfil_paciente)
    zona_val = ctx.perfil_paciente.zona.value

    if canal == CanalMonitoreo.TENS_INGRESA:
        modalidad   = "presencial_revision"
        costo_cons  = 10_380
        costo_viaje = _costo_viaje_zona(zona_val)
        t_paciente  = _tiempo_paciente_presencial_min(zona_val)
        t_especial  = 30
    else:
        modalidad   = "telemedicina_revision"
        costo_cons  = 8_630
        costo_viaje = 0
        t_paciente  = 20
        t_especial  = 25

    return {
        "paciente_id":             ctx.paciente_id,
        "control_num":             0,
        "tipo_evento":             "diagnostico_fase1_revision",
        "modalidad":               modalidad,
        "via_clinica":             "ninguna",
        "costo_consulta_clp":      costo_cons,
        "costo_viaje_clp":         costo_viaje,
        "costo_clp":               costo_cons + costo_viaje,
        "tiempo_paciente_min":     t_paciente,
        "tiempo_especialista_min": t_especial,
    }

def simular_paciente_longitudinal(
    ctx_inicial,
    pipeline_ews,
    modelo_evolucion,
    rng_eventos,
    n_controles: int = 4,
    nsp_activo: bool = False,
    predictor_nsp=None,
    perfil_nsp=None,
    historial_inasist: int = 0,
) -> tuple:
    from core.paciente_context import PacienteContext

    eventos      = []
    vias_control = []
    transiciones = []

    if not ctx_inicial.dm2_confirmado:
        zona_dx = ctx_inicial.perfil_paciente.zona.value
        eventos.append({
            "paciente_id":             ctx_inicial.paciente_id,
            "control_num":             0,
            "tipo_evento":             "diagnostico_fase1_ingreso",
            "modalidad":               "presencial_diagnostico",
            "via_clinica":             "ninguna",
            "costo_clp":               10_380,
            "tiempo_paciente_min":     _tiempo_paciente_presencial_min(zona_dx),
            "tiempo_especialista_min": 30,
        })
        eventos.append(_evento_revision_fase1(ctx_inicial))
        return eventos, [], []

    zona_dx = ctx_inicial.perfil_paciente.zona.value
    dia_ingreso = rng_eventos.uniform(0, 90)
    eventos.append({
        "paciente_id":             ctx_inicial.paciente_id,
        "control_num":             0,
        "dia":                     round(dia_ingreso),
        "tipo_evento":             "diagnostico_fase1_ingreso",
        "modalidad":               "presencial_diagnostico",
        "via_clinica":             "ninguna",
        "costo_clp":               10_380,
        "tiempo_paciente_min":     _tiempo_paciente_presencial_min(zona_dx),
        "tiempo_especialista_min": 30,
    })
    _rev = _evento_revision_fase1(ctx_inicial)
    _rev["dia"] = round(dia_ingreso + 10)
    eventos.append(_rev)

    datos_actuales  = ctx_inicial.datos_clinicos.copy()
    via_previa_str  = ctx_inicial.via_ruteo.value.upper() if ctx_inicial.via_ruteo else "AMARILLA"

    dia_control = dia_ingreso + 90
    control_num = 0
    while dia_control <= 365:
        control_num += 1
        _ctx_for_pool = ctx_inicial

        datos_previos  = datos_actuales.copy()
        datos_actuales = modelo_evolucion.evolucionar_paciente(
            datos_clinicos_previos=datos_actuales,
            via_previa=via_previa_str,
            control_num=control_num,
            paciente_id=ctx_inicial.paciente_id,
        )

        ctx_control = PacienteContext(
            paciente_id=ctx_inicial.paciente_id,
            perfil_paciente=ctx_inicial.perfil_paciente,
            fase_actual=FaseActual.FASE_2_EWS,
            dm2_confirmado=True,
            datos_clinicos=datos_actuales,
            especialista_titular=ctx_inicial.especialista_titular,
            fecha_proximo_control_rutina=ctx_inicial.fecha_proximo_control_rutina,
        )
        try:
            ctx_control = pipeline_ews.procesar(
                ctx_control,
                evento=f"control_trimestral_{control_num}",
            )
            _ctx_for_pool = ctx_control
            via_actual = (ctx_control.via_ruteo.value.upper()
                          if ctx_control.via_ruteo else "AMARILLA")
        except Exception:
            via_actual = via_previa_str

        if via_previa_str != via_actual:
            transiciones.append({
                "paciente_id": ctx_inicial.paciente_id,
                "control_num": control_num,
                "via_previa":  via_previa_str,
                "via_actual":  via_actual,
                "delta_hba1c": round(
                    datos_actuales["hba1c"] - datos_previos["hba1c"], 3
                ),
            })

        pool_tipo = getattr(_ctx_for_pool, "pool_tipo_asignacion", "nativo")

        vias_control.append({
            "paciente_id": ctx_inicial.paciente_id,
            "control_num": control_num,
            "via_clinica": via_actual,
            "pool_tipo":   pool_tipo,
            "hba1c":       datos_actuales.get("hba1c", 0),
            "adherencia":  datos_actuales.get("adherencia_tratamiento", 0),
        })

        ev_control = generar_evento_de_control(
            ctx_inicial.paciente_id, control_num, via_actual,
            ctx_inicial.perfil_paciente, rng_eventos,
        )

        if pool_tipo == "saturado" and via_actual in ("AMARILLA", "ROJA"):
            costo_sat = _costo_saturacion_pool(via_actual, ctx_inicial.perfil_paciente)
            eventos.append({
                "paciente_id":             ctx_inicial.paciente_id,
                "control_num":             control_num,
                "dia":                     round(dia_control),
                "tipo_evento":             "reprogramacion_pool_saturado",
                "modalidad":               "pool_saturado",
                "via_clinica":             via_actual,
                "costo_consulta_clp":      0,
                "costo_viaje_clp":         costo_sat - COSTO_REPROGRAMACION_ADMIN_CLP,
                "costo_clp":               costo_sat,
                "tiempo_paciente_min":     0,
                "tiempo_especialista_min": 0,
            })
            via_previa_str = via_actual
            dia_control += 90
            continue

        p_noshow = P_INASISTENCIA_VIA_EWS.get(via_actual, 0.10)
        iba_a_faltar = rng_eventos.random() < p_noshow

        if (iba_a_faltar and nsp_activo and predictor_nsp is not None
                and via_actual in ("AMARILLA", "ROJA")):
            _perf = perfil_nsp or {}
            _zona = _perf.get("zona_geografica", "URBANO")
            _score = predictor_nsp.calcular_score({
                "paciente_id":             ctx_inicial.paciente_id,
                "historial_inasistencias": historial_inasist,
                "via_clinica":             via_actual,
                "modalidad":               ev_control.get("modalidad", ""),
                "hba1c_actual":            datos_actuales.get("hba1c", 7.5),
                "zona_geografica":         _zona,
                "tramo_fonasa":            _perf.get("tramo_fonasa", "B"),
                "edad":                    _perf.get("edad", 60),
                "acceso_internet":         _perf.get("acceso_internet", True),
            })
            _, _, _costo_interv, _reduccion = predictor_nsp.decidir_intervencion(
                _score, _zona, via_clinica=via_actual)
            eventos.append({
                "paciente_id":             ctx_inicial.paciente_id,
                "control_num":             control_num,
                "dia":                     round(dia_control),
                "tipo_evento":             "intervencion_nsp",
                "modalidad":               "intervencion_nsp",
                "via_clinica":             via_actual,
                "costo_clp":               _costo_interv,
                "tiempo_paciente_min":     0,
                "tiempo_especialista_min": 0,
            })
            if predictor_nsp._rng.random() < _reduccion:
                iba_a_faltar = False

        if iba_a_faltar:
            costo_ins = ev_control["costo_consulta_clp"] + COSTO_REPROGRAMACION_ADMIN_CLP
            eventos.append({
                "paciente_id":             ctx_inicial.paciente_id,
                "control_num":             control_num,
                "dia":                     round(dia_control),
                "tipo_evento":             "inasistencia_ews",
                "modalidad":               "inasistencia",
                "via_clinica":             via_actual,
                "costo_consulta_clp":      ev_control["costo_consulta_clp"],
                "costo_viaje_clp":         0,
                "costo_clp":               costo_ins,
                "tiempo_paciente_min":     0,
                "tiempo_especialista_min": 0,
            })
            datos_actuales["adherencia_tratamiento"] = (
                datos_actuales.get("adherencia_tratamiento", 0.75) * 0.5)
            historial_inasist += 1
            via_previa_str = via_actual
            dia_control += 90
            continue

        ev_control["dia"] = round(dia_control)
        eventos.append(ev_control)
        datos_actuales["adherencia_tratamiento"] = min(
            1.0, datos_actuales.get("adherencia_tratamiento", 0.75) + 0.1)
        via_previa_str = via_actual
        dia_control += 90

    return eventos, vias_control, transiciones

def simular_escenario_solo_ews(
    n_pacientes: int = 500,
    dias_simulacion: int = 365,
    seed: int = 42,
    fuente_parametros: str = "asumido_literatura",
    generar_eventos_completos: bool = True,
    seed_sufijo: str = "",
    nsp_activo: bool = False,
) -> dict:
    sufijo = "ews_nsp" if nsp_activo else "solo_ews"
    print(f"\nsimulando {sufijo} | {n_pacientes} pacientes | {dias_simulacion} días")

    np.random.seed(seed)
    rng_eventos = _random.Random(seed + 1000)

    from core.evolucion_clinica import ModeloEvolucionClinica
    modelo_evolucion = ModeloEvolucionClinica(seed=seed + 2000)

    params = cargar_parametros_logisticos(fuente_parametros)

    red = RedEspecialistas()
    set_red_especialistas(red)

    cola_hitl          = ColaRevisionHITL()
    motor_escalamiento = MotorEscalamientoHITL(cola_hitl)

    from pipeline.fase2_ews import EWSPipeline
    pipeline_ews = EWSPipeline(cola_hitl=cola_hitl)

    df_pacientes        = pd.read_csv(RAIZ / "data/processed/pacientes_con_features.csv")
    pacientes_disponibles = df_pacientes["paciente_id"].unique()

    lote_pacientes = construir_lote_pacientes_ews(
        pacientes_disponibles=pacientes_disponibles,
        df_pacientes=df_pacientes,
        n_pacientes=n_pacientes,
        params=params,
        seed=seed,
        dias_simulacion=dias_simulacion,
    )

    from servicios.simulador_aprobacion_hitl import SimuladorAprobacionHITL
    sim_humano = SimuladorAprobacionHITL(seed=seed)

    metricas = {
        "escenario":                     sufijo,
        "total_procesados":              0,
        "fase1_confirmados":             0,
        "fase1_no_confirmados":          0,
        "fase2_via_verde":               0,
        "fase2_via_amarilla":            0,
        "fase2_via_roja":                0,
        "barrera_duras_bypass":          0,
        "uso_fallback_red":              0,
        "fallback_fallo_red_saturada":   0,
        "hitl_encolado":                 0,
        "hitl_aprobado_normal":          0,
        "hitl_escalado_t48":             0,
        "hitl_escalado_t72":             0,
        "hitl_escalado_t96_emergencia":  0,
        "control_rutina_adelantado":     0,
        "aprobadas_por_humano":              0,
        "rechazadas_por_humano":             0,
        "aprobadas_en_sla_normal":           0,
        "aprobadas_en_fase_a":               0,
        "aprobadas_en_fase_b":               0,
        "pacientes_reclasificados_post_hitl": 0,
        "transiciones_totales":                0,
        "pacientes_con_via_verde_alguna_vez":  0,
        "pacientes_que_mejoraron_de_via":      0,
        "pacientes_que_empeoraron_de_via":     0,
        "meses_promedio_perfil_completo":          0,
        "pacientes_logisticamente_vulnerables":    0,
        "distribucion_tramo_fonasa":               {"A": 0, "B": 0, "C": 0, "D": 0},
        "pacientes_sin_internet_domicilio":        0,
        "pacientes_canal_app_asignado":            0,
        "pacientes_canal_tens_asignado":           0,
        "pacientes_canal_whatsapp_asignado":       0,
    }

    eventos_simulacion    = []
    decisiones_simulacion = []
    eventos_completos        = []
    contextos_procesados     = []
    todas_las_vias_control   = []
    todas_las_transiciones   = []
    base_time                = datetime.now()

    for dia in range(dias_simulacion):
        actualizar_disponibilidad_red(red, dia, params)

        pacientes_del_dia = filtrar_pacientes_dia(lote_pacientes, dia, dias_simulacion)

        ahora_dia = base_time + timedelta(days=dia)

        for kwargs_paciente in pacientes_del_dia:
            try:
                ctx = procesar_paciente(pipeline_ews=pipeline_ews, **kwargs_paciente)

                metricas["total_procesados"] += 1
                actualizar_metricas_paciente(metricas, ctx)
                eventos_simulacion.append(extraer_evento_paciente(ctx, dia))

                if ctx.requiere_hitl and ctx.hitl_solicitud_id:
                    sol = cola_hitl.obtener(ctx.hitl_solicitud_id)
                    if sol:
                        sol.creada_en = ahora_dia

                for d in ctx.auditoria:
                    decisiones_simulacion.append({
                        "paciente_id":   ctx.paciente_id,
                        "dia_simulacion":dia,
                        "fase":          d.fase,
                        "motor":         d.motor,
                        "accion":        d.accion,
                    })

                contextos_procesados.append(ctx)

            except Exception as e:
                _log.warning(f"error en paciente {kwargs_paciente.get('paciente_id', '?')}: {e}")

        if sim_humano is not None:
            for evento_h in sim_humano.procesar_dia(cola_hitl, dia, ahora=ahora_dia):
                if evento_h.accion == "aprobada":
                    metricas["aprobadas_por_humano"] += 1
                    if evento_h.fase_protocolo == "SLA_normal":
                        metricas["aprobadas_en_sla_normal"] += 1
                    elif evento_h.fase_protocolo == "Fase_A_alerta":
                        metricas["aprobadas_en_fase_a"] += 1
                    elif evento_h.fase_protocolo == "Fase_B_respaldo":
                        metricas["aprobadas_en_fase_b"] += 1
                elif evento_h.accion == "rechazada":
                    metricas["rechazadas_por_humano"] += 1

        for ev in motor_escalamiento.procesar_escalamientos(ahora=ahora_dia):
            if ev.fase_protocolo == "A_preventiva":
                metricas["hitl_escalado_t48"] += 1
            elif ev.fase_protocolo == "B_pool":
                metricas["hitl_escalado_t72"] += 1
            elif ev.fase_protocolo == "C_emergencia":
                metricas["hitl_escalado_t96_emergencia"] += 1

        liberar_cupos_diarios(red, dia, params)

    if generar_eventos_completos:
        predictor_nsp   = None
        historial_prior = {}
        _perfil_nsp_fn  = None
        if nsp_activo:
            from servicios.predictor_nsp import PredictorNSP
            from importlib import import_module
            _perfil_nsp_fn = import_module("12_simulador_nsp")._perfil_sintetico
            predictor_nsp  = PredictorNSP(seed=seed)
            _rng_prior     = np.random.default_rng(seed + 9999)
            for _c in contextos_procesados:
                _pp  = _perfil_nsp_fn(_c.paciente_id, seed)
                _lam = 0.60
                if _pp["tramo_fonasa"] == "A":
                    _lam += 0.20
                if "RURAL" in _pp["zona_geografica"]:
                    _lam += 0.15
                if not _pp["acceso_internet"]:
                    _lam += 0.10
                _pr = int(_rng_prior.poisson(_lam))
                if _pr > 0:
                    historial_prior[_c.paciente_id] = _pr

        for ctx in contextos_procesados:
            if (sim_humano.es_reclasificado(ctx.paciente_id)
                    and ctx.via_ruteo == ViaRuteo.VERDE):
                ctx.via_ruteo = ViaRuteo.AMARILLA
                metricas["pacientes_reclasificados_post_hitl"] += 1

            red.resetear_cargas_diarias()

            evs, vias_p, trans_p = simular_paciente_longitudinal(
                ctx_inicial=ctx,
                pipeline_ews=pipeline_ews,
                modelo_evolucion=modelo_evolucion,
                rng_eventos=rng_eventos,
                nsp_activo=nsp_activo,
                predictor_nsp=predictor_nsp,
                perfil_nsp=(_perfil_nsp_fn(ctx.paciente_id, seed)
                            if _perfil_nsp_fn else None),
                historial_inasist=historial_prior.get(ctx.paciente_id, 0),
            )
            eventos_completos.extend(evs)
            todas_las_vias_control.extend(vias_p)
            todas_las_transiciones.extend(trans_p)

            if any(v["via_clinica"] == "VERDE" for v in vias_p):
                metricas["pacientes_con_via_verde_alguna_vez"] += 1

        if todas_las_transiciones:
            metricas["transiciones_totales"] = len(todas_las_transiciones)
            tipos: dict = {}
            for t in todas_las_transiciones:
                clave = f"{t['via_previa']}_a_{t['via_actual']}"
                tipos[clave] = tipos.get(clave, 0) + 1
            metricas["transiciones_por_tipo"] = tipos

            for t in todas_las_transiciones:
                mejora = (
                    (t["via_previa"] == "ROJA"    and t["via_actual"] in ("AMARILLA", "VERDE")) or
                    (t["via_previa"] == "AMARILLA" and t["via_actual"] == "VERDE")
                )
                empeora = (
                    (t["via_previa"] == "VERDE"    and t["via_actual"] in ("AMARILLA", "ROJA")) or
                    (t["via_previa"] == "AMARILLA" and t["via_actual"] == "ROJA")
                )
                if mejora:
                    metricas["pacientes_que_mejoraron_de_via"] += 1
                elif empeora:
                    metricas["pacientes_que_empeoraron_de_via"] += 1

    ctxs_con_calendario = [c for c in contextos_procesados
                           if c.calendario_examenes is not None]
    if ctxs_con_calendario:
        meses_total = sum(
            c.calendario_examenes.meses_hasta_perfil_completo
            for c in ctxs_con_calendario
        )
        metricas["meses_promedio_perfil_completo"] = round(
            meses_total / len(ctxs_con_calendario), 2
        )

    _log.info(
        f"ews listo | {n_pacientes} pacientes | {dias_simulacion} días | "
        f"procesados={metricas['total_procesados']} "
        f"confirmados={metricas['fase1_confirmados']} "
        f"HITL={metricas['hitl_encolado']} "
        f"reclasificados={metricas['pacientes_reclasificados_post_hitl']}"
    )

    if generar_eventos_completos and eventos_completos:
        df_ev = pd.DataFrame(eventos_completos)
        metricas["eventos_completos_total"]      = len(df_ev)
        metricas["costo_eventos_total_clp"]      = int(df_ev["costo_clp"].sum())
        metricas["tiempo_paciente_total_min"]    = int(df_ev["tiempo_paciente_min"].sum())
        metricas["tiempo_especialista_total_min"]= int(df_ev["tiempo_especialista_min"].sum())
        for mod, n in df_ev["tipo_evento"].value_counts().to_dict().items():
            metricas[f"eventos_{mod}"] = int(n)
        df_ev.to_csv(
            DIR_REPORTS / f"simulacion_{sufijo}_eventos_completos{seed_sufijo}.csv",
            index=False,
        )

        import json
        if todas_las_vias_control:
            df_vias = pd.DataFrame(todas_las_vias_control)
            df_vias.to_csv(
                DIR_REPORTS / f"simulacion_{sufijo}_vias_por_control{seed_sufijo}.csv",
                index=False,
            )
            if "pool_tipo" in df_vias.columns:
                for tipo, cnt in df_vias["pool_tipo"].value_counts().to_dict().items():
                    metricas[f"pool_{tipo}"] = int(cnt)
        if todas_las_transiciones:
            pd.DataFrame(todas_las_transiciones).to_csv(
                DIR_REPORTS / f"simulacion_{sufijo}_transiciones{seed_sufijo}.csv",
                index=False,
            )
        stats_evol = modelo_evolucion.estadisticas_evolucion()
        with open(
            DIR_REPORTS / f"simulacion_{sufijo}_evolucion{seed_sufijo}.json", "w",
            encoding="utf-8",
        ) as f:
            json.dump(stats_evol, f, indent=2, ensure_ascii=False)

    guardar_resultados_ews(metricas, eventos_simulacion, decisiones_simulacion,
                           cola_hitl, motor_escalamiento, sufijo=sufijo,
                           seed_sufijo=seed_sufijo)

    if sim_humano.eventos:
        df_humano = pd.DataFrame([
            {
                "solicitud_id":       e.solicitud_id,
                "paciente_id":        e.paciente_id,
                "accion":             e.accion,
                "fase_protocolo":     e.fase_protocolo,
                "horas_transcurridas":e.horas_transcurridas,
                "dia_simulacion":     e.dia_simulacion,
                "medico_id":          e.medico_id,
            }
            for e in sim_humano.eventos
        ])
        df_humano.to_csv(
            DIR_REPORTS / f"simulacion_{sufijo}_aprobaciones_humanas{seed_sufijo}.csv",
            index=False,
        )

    set_red_especialistas(None)
    return metricas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador DES - diseno factorial 2x2")
    parser.add_argument("--escenario",   choices=["base", "solo_ews", "ews_nsp"])
    parser.add_argument("--n-pacientes", type=int, default=500)
    parser.add_argument("--dias",        type=int, default=365)
    parser.add_argument("--fuente",      default="asumido_literatura")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-replicas",  type=int, default=1,
                        help="Numero de replicas independientes con semillas "
                             "seed, seed+1, ..., seed+n-1. Default=1 (sin sufijo).")
    args = parser.parse_args()

    semillas = [args.seed + i for i in range(args.n_replicas)]

    for idx, semilla in enumerate(semillas):
        if args.n_replicas > 1:
            print(f"\n[Replica {idx+1}/{args.n_replicas}] seed={semilla}")
        seed_suf = f"_seed{semilla}" if args.n_replicas > 1 else ""

        if args.escenario in ("solo_ews", "ews_nsp"):
            metricas = simular_escenario_solo_ews(
                n_pacientes=args.n_pacientes,
                dias_simulacion=args.dias,
                seed=semilla,
                fuente_parametros=args.fuente,
                seed_sufijo=seed_suf,
                nsp_activo=(args.escenario == "ews_nsp"),
            )
            print(f"\nResumen {args.escenario} (seed={semilla})")
            total = metricas["total_procesados"] or 1
            for k, v in metricas.items():
                if isinstance(v, int):
                    print(f"  {k:<40}: {v:>6}  ({round(100*v/total,1):.1f}%)")
            if "costo_eventos_total_clp" in metricas:
                costo_ev = metricas["costo_eventos_total_clp"]
                t_pac    = metricas["tiempo_paciente_total_min"]
                n_verde  = metricas.get("pacientes_con_via_verde_alguna_vez", 0)
                n_conf   = metricas["fase1_confirmados"] or 1
                print(f"\n  costo_eventos_total_clp  : ${costo_ev:>15,.0f}")
                print(f"  tiempo_paciente_total_min: {t_pac:>6,.0f} min ({t_pac/60:.0f}h)")
                print(f"  pacientes_via_verde_algun_ctrl : {n_verde} ({100*n_verde/n_conf:.1f}%)")
                print(f"  transiciones_totales     : {metricas.get('transiciones_totales',0)}")

        elif args.escenario == "base":
            params = cargar_parametros(args.fuente)
            pacientes, log_ev = correr_simulacion("base", args.n_pacientes,
                                                  args.dias, params, semilla)
            resumen = generar_reportes(pacientes, log_ev, "base",
                                       seed_sufijo=seed_suf)
            print(f"\nResumen base (seed={semilla})")
            for k, v in resumen.items():
                print(f"  {k:<30}: {f'{v:,}' if isinstance(v, int) else v}")

        else:
            parser.print_help()
            break
