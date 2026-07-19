set -euo pipefail

N_REPLICAS="${N_REPLICAS:-30}"
SEED="${SEED:-42}"

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYROOT="$RAIZ/sistema_tratamientos/sistema_tratamientos"
PY="$RAIZ/venv/bin/python"

export PYTHONIOENCODING="utf-8"

if [[ ! -x "$PY" ]]; then
    echo "No se encontró el venv en $PY" >&2
    exit 1
fi
cd "$PYROOT"

step() {
    echo ""
    echo "[$(date +%H:%M:%S)] PASO $1 - $2"
}
step  1 "02c - Genera dataset sintético (seed=$SEED)"
"$PY" src/02c_generate_data.py
step  2 "03  - Limpieza con flags de faltantes"
"$PY" src/03_clean.py
step  3 "04  - Feature engineering (90 variables)"
"$PY" src/04_features.py
step  4 "05  - Split temporal por paciente (70/15/15)"
"$PY" src/05_split.py
step  5 "06a - Entrena XGBoost + benchmark"
"$PY" src/06a_train_ml.py
step  6 "06c - Tablas 9 y 10 (comparación de modelos)"
"$PY" src/06c_compare_models.py
step  7 "07  - Métricas finales + SHAP global"
"$PY" src/07_evaluate.py
step  8 "07b - Tabla 14 (pacientes ancla SHAP)"
"$PY" src/07b_shap_global.py
step  9 "07c - Tabla 12 (Decision Curve Analysis)"
"$PY" src/07c_decision_curve_analysis.py

step 10 "Simulación 2x2 - $N_REPLICAS réplicas (seeds $SEED..$((SEED + N_REPLICAS - 1)))"
"$PY" scripts/run_replicas_2x2.py --n-replicas "$N_REPLICAS" --seed "$SEED"

step 11 "13  - Tabla 19 + sub-aditividad"
"$PY" src/13_consolidar_2x2.py --n-replicas "$N_REPLICAS" --seed "$SEED"
step 12 "14  - Tablas 21, 23, 24, 25, 26 (Anexo D)"
"$PY" src/14_consolidar_anexo_d.py --n-replicas "$N_REPLICAS" --seed "$SEED"

echo ""
echo "Resultados en reports/ y reports/evaluacion_final/."