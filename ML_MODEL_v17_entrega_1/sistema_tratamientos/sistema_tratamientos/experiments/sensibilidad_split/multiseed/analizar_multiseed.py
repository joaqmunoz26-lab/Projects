from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DIR_EXP     = Path(__file__).resolve().parent
DIR_RESULTS = DIR_EXP / "results"

def analizar():
    ruta = DIR_RESULTS / "metricas_multiseed.csv"
    df = pd.read_csv(ruta)
    print("=" * 65)
    print("Análisis multi semilla: distribuciones y tests estadísticos")
    print(f"Filas: {len(df)} | Semillas: {sorted(df['semilla'].unique())}")
    print("=" * 65)

    metricas_cmp = ["auc_roc", "auc_pr", "f1"]
    modelos = ["XGBoost", "Random Forest"]
    estrategias = ["aleatorio", "estratificado"]

    print("\n[1] Distribuciones (media +- SD, min-max)")
    filas_dist = []
    for m in modelos:
        for e in estrategias:
            sub = df[(df["modelo"] == m) & (df["estrategia"] == e)]
            for met in metricas_cmp:
                vals = sub[met].values
                fila = {
                    "modelo": m, "estrategia": e, "metrica": met,
                    "media": round(vals.mean(), 4),
                    "sd":    round(vals.std(ddof=1), 4),
                    "min":   round(vals.min(), 4),
                    "max":   round(vals.max(), 4),
                    "n":     len(vals),
                }
                filas_dist.append(fila)
                print(
                    f"  {m:<14} {e:<14} {met:<8} "
                    f"{vals.mean():.4f} +- {vals.std(ddof=1):.4f}  "
                    f"[{vals.min():.4f}, {vals.max():.4f}]"
                )

    df_dist = pd.DataFrame(filas_dist)
    df_dist.to_csv(DIR_RESULTS / "analisis_multiseed.csv", index=False)
    print(f"\n  -> Guardado: {DIR_RESULTS / 'analisis_multiseed.csv'}")
    print("\n[2] Solapamiento: semillas donde estratificado > aleatorio")
    for m in modelos:
        for met in metricas_cmp:
            ale = df[(df["modelo"] == m) & (df["estrategia"] == "aleatorio")].set_index("semilla")[met]
            est = df[(df["modelo"] == m) & (df["estrategia"] == "estratificado")].set_index("semilla")[met]
            comun = ale.index.intersection(est.index)
            n_sup = (est[comun] > ale[comun]).sum()
            deltas = (est[comun] - ale[comun]).values
            print(
                f"  {m:<14} {met:<8} "
                f"est>ale en {n_sup}/{len(comun)} semillas  "
                f"delta medio {deltas.mean():+.4f}"
            )

    print("\n[3] Test de Wilcoxon signed-rank pareado por semilla (n=10 pares)")
    filas_test = []
    for m in modelos:
        for met in metricas_cmp:
            ale = df[(df["modelo"] == m) & (df["estrategia"] == "aleatorio")].set_index("semilla")[met]
            est = df[(df["modelo"] == m) & (df["estrategia"] == "estratificado")].set_index("semilla")[met]
            comun = ale.index.intersection(est.index)
            diffs = (est[comun] - ale[comun]).values
            if np.all(diffs == 0):
                stat, pval = np.nan, np.nan
            else:
                stat, pval = wilcoxon(diffs, alternative="two-sided")
            filas_test.append({
                "modelo": m, "metrica": met,
                "delta_medio": round(diffs.mean(), 4),
                "delta_sd":    round(diffs.std(ddof=1), 4),
                "W_stat": round(stat, 2) if not np.isnan(stat) else None,
                "p_valor": round(pval, 4) if not np.isnan(pval) else None,
                "significativo_0.05": (pval < 0.05) if not np.isnan(pval) else None,
            })
            sig = "p<0.05 *" if (not np.isnan(pval) and pval < 0.05) else "n.s."
            print(
                f"  {m:<14} {met:<8} "
                f"delta={diffs.mean():+.4f} +- {diffs.std(ddof=1):.4f}  "
                f"W={stat:.0f}  p={pval:.4f}  {sig}"
            )

    pd.DataFrame(filas_test).to_csv(DIR_RESULTS / "wilcoxon_multiseed.csv", index=False)

    print("\n[4] Inversión de ranking (XGBoost vs RF, por estrategia y semilla)")
    filas_rank = []
    for e in estrategias:
        sub_e = df[df["estrategia"] == e]
        for met in metricas_cmp:
            xgb_wins = 0
            rf_wins  = 0
            ties     = 0
            for s in sorted(df["semilla"].unique()):
                sub_s = sub_e[sub_e["semilla"] == s]
                val_xgb = sub_s[sub_s["modelo"] == "XGBoost"][met].values
                val_rf  = sub_s[sub_s["modelo"] == "Random Forest"][met].values
                if len(val_xgb) == 0 or len(val_rf) == 0:
                    continue
                v_x, v_r = val_xgb[0], val_rf[0]
                if v_x > v_r:
                    xgb_wins += 1
                elif v_r > v_x:
                    rf_wins  += 1
                else:
                    ties += 1
            filas_rank.append({
                "estrategia": e, "metrica": met,
                "xgboost_gana": xgb_wins,
                "rf_gana": rf_wins,
                "empate": ties,
            })
            print(
                f"  {e:<14} {met:<8} "
                f"XGB gana: {xgb_wins}/10  RF gana: {rf_wins}/10  "
                f"empate: {ties}/10"
            )

    df_rank = pd.DataFrame(filas_rank)
    df_rank.to_csv(DIR_RESULTS / "ranking_multiseed.csv", index=False)
    print(f"\n Guardado: {DIR_RESULTS / 'ranking_multiseed.csv'}")

    print("\n[5] Dispersión de prevalencias train/val/test por estrategia")
    for e in ["aleatorio", "estratificado"]:
        sub = df[df["estrategia"] == e].drop_duplicates(subset=["semilla", "estrategia"])
        if sub.empty:
            sub = df[(df["estrategia"] == e) & (df["modelo"] == "XGBoost")]
        disp = sub.apply(
            lambda r: np.std([r["prev_train"], r["prev_val"], r["prev_test"]], ddof=0),
            axis=1,
        )
        print(
            f"  {e:<14} dispersion prev (SD entre 3 partic): "
            f"media={disp.mean():.4f}  max={disp.max():.4f}"
        )

    print("\n" + "=" * 65)
    print("Síntesis de métricas")
    print("=" * 65)
    for m in modelos:
        ale_auc = df[(df["modelo"] == m) & (df["estrategia"] == "aleatorio")]["auc_roc"]
        est_auc = df[(df["modelo"] == m) & (df["estrategia"] == "estratificado")]["auc_roc"]
        comun = ale_auc.index.intersection(est_auc.index)
        n_pos = (est_auc.values > ale_auc.values).sum()
        print(
            f"  {m}: estratificado > aleatorio en {n_pos}/10 semillas "
            f"(AUC-ROC delta medio {(est_auc.values - ale_auc.values).mean():+.4f})"
        )

if __name__ == "__main__":
    analizar()
