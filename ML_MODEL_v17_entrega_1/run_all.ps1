param(
    [int]$NReplicas = 30,
    [int]$Seed      = 42
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING   = "utf-8"

$Raiz   = $PSScriptRoot
$PyRoot = Join-Path $Raiz "sistema_tratamientos\sistema_tratamientos"
$Py     = Join-Path $Raiz "venv\Scripts\python.exe"

if (-not (Test-Path $Py)) { throw "No se encontro el venv en $Py." }
Set-Location $PyRoot

function Step($desc, $ScriptArgs) {
    Write-Host "`n[$(Get-Date -Format HH:mm:ss)] $desc" -ForegroundColor Cyan
    & $Py @ScriptArgs
    if ($LASTEXITCODE -ne 0) { throw "Fallo: $desc" }
}

Step "02c - Genera dataset sintetico (seed=$Seed)"      @("src/02c_generate_data.py")
Step "03  - Limpieza con flags de faltantes"            @("src/03_clean.py")
Step "04  - Feature engineering (90 variables)"         @("src/04_features.py")
Step "05  - Split temporal por paciente (70/15/15)"     @("src/05_split.py")
Step "06a - Entrena XGBoost + benchmark"                @("src/06a_train_ml.py")
Step "06c - Tablas 9 y 10 (comparacion de modelos)"     @("src/06c_compare_models.py")
Step "07  - Metricas finales + SHAP global"             @("src/07_evaluate.py")
Step "07b - Tabla 14 (pacientes ancla SHAP)"            @("src/07b_shap_global.py")
Step "07c - Tabla 12 (Decision Curve Analysis)"         @("src/07c_decision_curve_analysis.py")

Step "Simulacion 2x2 - $NReplicas replicas (seeds $Seed..$($Seed+$NReplicas-1))" `
     @("scripts/run_replicas_2x2.py", "--n-replicas", "$NReplicas", "--seed", "$Seed")

Step "13  - Tabla 19 + sub-aditividad"                  @("src/13_consolidar_2x2.py", "--n-replicas", "$NReplicas", "--seed", "$Seed")
Step "14  - Tablas 21, 23, 24, 25, 26 (Anexo D)"        @("src/14_consolidar_anexo_d.py", "--n-replicas", "$NReplicas", "--seed", "$Seed")

Write-Host "`nResultados en reports/ y reports/evaluacion_final/." -ForegroundColor Green