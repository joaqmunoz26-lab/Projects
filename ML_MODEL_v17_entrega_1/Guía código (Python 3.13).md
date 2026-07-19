**Guía código (Python 3.13)**



Proyecto en ML_MODEL_v17_entrega_1/sistema_tratamientos/sistema_tratamientos/. Venv en ML_MODEL_v17_entrega_1/venv/.



**Archivos principales**

scripts/run_replicas_2x2.py: experimento 2×2.

src/10_simulador_eventos.py: escenarios base, solo_ews y ews_nsp.

src/12_simulador_nsp.py: escenario solo_nsp.

src/13_consolidar_2x2.py: réplicas, costos, ahorros con IC 95 %.

src/motor/motor_ml.py: modelo predictivo con XGBoost.

src/motor/reglas_duras.py: reglas clínicas previas al uso de XGBoost.

src/servicios/predictor_nsp.py: Score heurístico de inasistencias (NSP).

src/servicios/hitl_revision.py: Revisión humana y escalamiento (SLA/HITL).

00_parametros_logisticos.yaml: Costos y tiempos del modelo.



**Instalación**

Desde ML_MODEL_v17_entrega_1/ (crea el venv en ML_MODEL_v17_entrega_1/venv/):

python -m venv venv

venv\Scripts\Activate.ps1



**Correr con un solo comando**

(Desde ML_MODEL_v17_entrega_1/):


.\run_all.ps1 #En Windows(defaults: 30 réplicas, seed 42)



**Correr paso a paso**

Desde sistema_tratamientos/sistema_tratamientos/ con venv activado.

**Comando 1:** python src/10_simulador_eventos.py --escenario base --n-replicas 30 --seed 42
**Comando 2:** python src/10_simulador_eventos.py --escenario solo_ews --n-replicas 30 --seed 42

**Comando 3**: python src/10_simulador_eventos.py --escenario ews_nsp --n-replicas 30 --seed 42

Comandos 1 a 3: generan los tres escenarios del diseño 2×2.

**Comando 4:** python src/12_simulador_nsp.py --n-replicas 30 --seed 42

Comando 4: genera el escenario solo_nsp.

**Comando 5:** python src/13_consolidar_2x2.py --n-replicas 30 --seed 42

Comando 5: consolida las réplicas y calcula costos y ahorros con IC 95 %.



**Salida:** reports/evaluacion_final/comparacion_2x2_costos.csv



**Verificar que el código funciona sin sobrescribir los CSV.**

python -m pytest -q



**NOTA: Al correr el código completo si se sobrescriben los CSV de seed 42. Los resultados deberían ser los mismos o casi idénticos al informe y/o presentación.**
