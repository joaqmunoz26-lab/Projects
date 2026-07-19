import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
_SRC  = _RAIZ / "src"

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _run(descripcion: str, cmd: list[str]) -> float:
    print(f"\n[{_ts()}] inicio - {descripcion}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[{_ts()}] fin - {descripcion}  ({elapsed:.1f}s / {elapsed/60:.1f} min)")
    return elapsed

def _mostrar_csv(ruta: Path) -> None:
    if ruta.exists():
        print(f"\n{ruta.name}")
        print(ruta.read_text(encoding="utf-8"))
    else:
        print(f"  [no encontrado] {ruta}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador de la corrida 2x2 con N replicas reproducibles."
    )
    parser.add_argument("--n-replicas",  type=int, default=30,
                        help="Numero de replicas (default: 30, semillas 42..71).")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Semilla inicial (default: 42).")
    parser.add_argument("--n-pacientes", type=int, default=500,
                        help="Pacientes por replica (default: 500).")
    parser.add_argument("--dias",        type=int, default=365,
                        help="Dias de simulacion por replica (default: 365).")
    args = parser.parse_args()

    py    = sys.executable
    n     = args.n_replicas
    seed  = args.seed
    npac  = args.n_pacientes
    dias  = args.dias

    print(f"\nOrquestador 2x2 - {n} réplicas (seeds {seed}-{seed+n-1})")
    print(f"  {npac} pacientes x {dias} días x 4 escenarios")
    print(f"  Inicio: {_ts()}")

    t_total = time.time()
    tiempos = {}

    tiempos["base"] = _run(
        f"escenario base ({n} réplicas)",
        [py, str(_SRC / "10_simulador_eventos.py"),
         "--escenario", "base",
         "--n-pacientes", str(npac),
         "--dias",        str(dias),
         "--seed",        str(seed),
         "--n-replicas",  str(n)],
    )

    tiempos["solo_ews"] = _run(
        f"escenario solo_ews ({n} réplicas)",
        [py, str(_SRC / "10_simulador_eventos.py"),
         "--escenario", "solo_ews",
         "--n-pacientes", str(npac),
         "--dias",        str(dias),
         "--seed",        str(seed),
         "--n-replicas",  str(n)],
    )

    tiempos["ews_nsp"] = _run(
        f"escenario ews_nsp ({n} réplicas)",
        [py, str(_SRC / "10_simulador_eventos.py"),
         "--escenario", "ews_nsp",
         "--n-pacientes", str(npac),
         "--dias",        str(dias),
         "--seed",        str(seed),
         "--n-replicas",  str(n)],
    )

    tiempos["solo_nsp"] = _run(
        f"capa NSP - solo_nsp ({n} réplicas)",
        [py, str(_SRC / "12_simulador_nsp.py"),
         "--seed",       str(seed),
         "--n-replicas", str(n)],
    )

    tiempos["consolidar"] = _run(
        "consolidador 2x2 + subaditividad",
        [py, str(_SRC / "13_consolidar_2x2.py"),
         "--n-replicas", str(n),
         "--seed",       str(seed)],
    )

    elapsed_total = time.time() - t_total

    print("\nTiempos por paso")
    for paso, t in tiempos.items():
        print(f"  {paso:<15}: {t:>7.1f}s  ({t/60:.1f} min)")
    print(f"  {'total':<15}: {elapsed_total:>7.1f}s  ({elapsed_total/60:.1f} min)")

    dir_final = _RAIZ / "reports" / "evaluacion_final"
    print("\n\nResultados finales:")
    _mostrar_csv(dir_final / "comparacion_2x2_costos.csv")
    _mostrar_csv(dir_final / "comparacion_2x2_subaditividad.csv")

if __name__ == "__main__":
    main()
