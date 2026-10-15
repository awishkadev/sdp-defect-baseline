"""Command line entry point.

    python run_baseline.py                 # full run: tuning + untuned + tuned, 30 repeats
    python run_baseline.py --quick         # fast smoke run (default parameters, 2 repeats, no tuning)
    python run_baseline.py --out results   # choose the output folder
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from sdp import experiments as ex
from sdp.config import N_REPEATS, PAIRS, PAPER


def summarise(cv, label):
    rows = []
    for m in ["All-buggy rule"] + [c[0] for c in ex.CONFIGS]:
        d = cv[cv.model == m]
        rows.append(dict(setting=label, model=m, F1=d.f1.mean(), AUC=d.auc.mean(),
                         PofB20=d.pofb20.mean(), PofB20_per_LOC=d.pofb20_per_loc.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--quick", action="store_true", help="smoke run: default parameters, 2 repeats, no tuning")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reps = 2 if a.quick else N_REPEATS
    t0 = time.time()

    if a.quick:
        # smoke run: default parameters only, 2 repeats (no tuning, no within-release check)
        print(f"[quick] cross-version, default parameters ({reps} repeats) ...")
        cv_def = ex.cross_version(None, n_repeats=reps)
        cv_def.to_csv(out / "results_cross_version_default.csv", index=False)
        print(summarise(cv_def, "default").round(3).to_string(index=False))
        print(f"\ndone in {time.time() - t0:.0f} s; files written to {out}/")
        return

    print("[1/4] tuning LR and RF on the 5 training releases ...")
    best, tune_table = ex.tune_all()
    tune_table.to_csv(out / "tuning_candidates.csv", index=False)
    for old, d in best.items():
        print(f"      {old:12s} best LR {d['LR']}  best RF {d['RF']}")

    print(f"[2/4] cross-version, default parameters ({reps} repeats) ...")
    cv_def = ex.cross_version(None, n_repeats=reps)
    print(f"[3/4] cross-version, tuned parameters ({reps} repeats) ...")
    cv_tuned = ex.cross_version(best, n_repeats=reps)
    cv_def.to_csv(out / "results_cross_version_default.csv", index=False)
    cv_tuned.to_csv(out / "results_cross_version_tuned.csv", index=False)

    summ = pd.concat([summarise(cv_def, "default"), summarise(cv_tuned, "tuned")])
    tests = [p[1] for p in PAIRS]
    paper = pd.DataFrame([PAPER[t] for t in tests], columns=["F1", "AUC", "PofB20"]).mean()
    summ = pd.concat([summ, pd.DataFrame([dict(setting="paper", model="CNN-MLP (paper)", F1=paper.F1,
                                               AUC=paper.AUC, PofB20=paper.PofB20, PofB20_per_LOC=float("nan"))])])
    summ.to_csv(out / "summary_vs_paper.csv", index=False)
    pd.set_option("display.width", 160)
    print("\nMean over the 5 test versions (paper = CNN-MLP as printed in the paper):")
    print(summ.round(3).to_string(index=False))

    print("[4/4] within-release 5-fold check for all 12 releases (default parameters) ...")
    wv = ex.within_version()
    wv.to_csv(out / "results_within_version.csv", index=False)
    print(wv.round(3).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f} s; files written to {out}/")


if __name__ == "__main__":
    main()
