"""Re-checks of the numbers printed in the paper (Abdu et al., 2024), used in the Phase 3 report.

    python check_paper_tables.py

The values of the paper's Tables 5 (F1), 6 (AUC) and 10 (PofB20) are typed in below, one row per
experiment, columns TR, CNN, DBN, LSTM, DP-HNN, SDP-BB, ACGDP, CNN-MLP. The script
  1. recomputes the column averages and compares them with the averages printed in the paper,
  2. lists values of Table 10 that are identical to the same cell of Table 6,
  3. tests CNN-MLP against each baseline across the 12 experiments (Wilcoxon signed-rank test)
     and applies Holm's correction for the 7 comparisons of one metric,
  4. compares the F1 of CNN-MLP with the "every class is buggy" rule (F1 = 2p / (1 + p)).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sdp.data import load_release

MODELS = ["TR", "CNN", "DBN", "LSTM", "DP-HNN", "SDP-BB", "ACGDP", "CNN-MLP"]
ROWS = ["xalan-2.4", "xalan-2.6", "poi-1.5", "poi-3.0", "ant-1.7", "log4j-1.1",
        "jedit-4.0", "jedit-4.1", "lucene-2.0", "lucene-2.2", "synapse-1.1", "synapse-1.2"]

# Table 5 (F1) in the order printed in the paper ...
_F1_ORDER = ["jedit-4.0", "jedit-4.1", "ant-1.7", "synapse-1.1", "synapse-1.2", "xalan-2.4",
             "xalan-2.6", "poi-1.5", "poi-3.0", "log4j-1.1", "lucene-2.0", "lucene-2.2"]
_F1 = [[.523, .620, .623, .616, .622, .651, .630, .675], [.389, .634, .477, .593, .630, .627, .661, .652],
       [.563, .621, .661, .711, .732, .715, .704, .719], [.504, .419, .474, .560, .534, .565, .523, .572],
       [.610, .654, .563, .688, .681, .673, .636, .690], [.618, .668, .684, .652, .640, .728, .667, .689],
       [.549, .643, .669, .644, .563, .673, .682, .706], [.724, .688, .748, .732, .791, .817, .816, .856],
       [.774, .801, .773, .700, .721, .700, .792, .745], [.622, .585, .498, .694, .733, .661, .704, .845],
       [.642, .636, .691, .589, .658, .606, .636, .704], [.609, .623, .709, .552, .527, .523, .562, .578]]
# ... and Tables 6 (AUC) and 10 (PofB20), which use the order of ROWS.
_AUC = [[.456, .562, .592, .633, .651, .683, .673, .712], [.512, .535, .552, .570, .635, .652, .620, .626],
        [.436, .553, .502, .546, .560, .534, .548, .566], [.532, .492, .473, .462, .583, .573, .597, .616],
        [.541, .601, .596, .658, .622, .631, .636, .673], [.476, .583, .565, .572, .540, .558, .570, .592],
        [.589, .511, .543, .503, .523, .541, .507, .552], [.488, .589, .534, .481, .501, .519, .593, .582],
        [.492, .584, .632, .590, .641, .605, .611, .637], [.514, .572, .579, .522, .566, .524, .572, .596],
        [.495, .499, .587, .583, .560, .595, .505, .553], [.595, .574, .592, .667, .620, .633, .600, .683]]
_POF = [[.456, .369, .612, .587, .603, .597, .605, .619], [.512, .435, .552, .611, .580, .616, .582, .626],
        [.436, .563, .574, .550, .569, .577, .584, .566], [.532, .563, .473, .487, .592, .482, .533, .492],
        [.241, .221, .296, .361, .305, .351, .366, .406], [.376, .386, .305, .372, .322, .374, .320, .415],
        [.401, .376, .343, .300, .310, .364, .361, .398], [.388, .371, .334, .339, .377, .371, .377, .389],
        [.462, .484, .598, .472, .582, .541, .609, .623], [.453, .514, .615, .624, .605, .636, .611, .612],
        [.295, .299, .287, .455, .411, .452, .426, .493], [.285, .274, .292, .480, .436, .467, .488, .521]]
PRINTED_AVG = {"F1": [.594, .633, .631, .644, .653, .662, .668, .703],
               "AUC": [.511, .555, .562, .566, .584, .587, .586, .616],
               "PofB20": [.403, .405, .440, .470, .474, .486, .489, .513]}


def tables():
    f1 = pd.DataFrame(_F1, index=_F1_ORDER, columns=MODELS).loc[ROWS]
    return {"F1": f1, "AUC": pd.DataFrame(_AUC, index=ROWS, columns=MODELS),
            "PofB20": pd.DataFrame(_POF, index=ROWS, columns=MODELS)}


def holm(p):
    """Holm's step-down adjustment of a list of p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    T = tables()
    print("1. Column averages recomputed from the typed tables (max difference to the printed average):")
    for k, df in T.items():
        print(f"   {k:7s} max |diff| = {np.abs(df.mean().to_numpy() - np.array(PRINTED_AVG[k])).max():.4f}")

    same = (T["AUC"] == T["PofB20"])
    print(f"\n2. Cells of Table 10 (PofB20) identical to the same cell of Table 6 (AUC): {int(same.to_numpy().sum())} of {same.size}")
    for r in ROWS:
        for m in MODELS:
            if same.loc[r, m]:
                print(f"   {r:10s} {m:8s} {T['AUC'].loc[r, m]:.3f}")

    print("\n3. CNN-MLP against each baseline across the 12 experiments (Wilcoxon signed-rank, two-sided):")
    for k, df in T.items():
        base = MODELS[:-1]
        p = np.array([wilcoxon(df["CNN-MLP"], df[b]).pvalue for b in base])
        adj = holm(p)
        print(f"   {k}: wins of 12 / mean difference / p / Holm-adjusted p")
        for b, pv, a in zip(base, p, adj):
            d = df["CNN-MLP"] - df[b]
            print(f"      vs {b:7s} {int((d > 0).sum()):2d}   {d.mean():+.3f}   {pv:.4f}   {a:.3f}")
        best = int((df.idxmax(axis=1) == "CNN-MLP").sum())
        print(f"      CNN-MLP has the best value in {best} of 12 experiments")

    print("\n4. F1 of CNN-MLP against the rule 'every class is buggy' (p = buggy share of the version):")
    gap = []
    for r in ROWS:
        p = load_release(r).y.mean()
        rule = 2 * p / (1 + p)
        gap.append(T["F1"].loc[r, "CNN-MLP"] - rule)
        print(f"   {r:12s} p = {p:.3f}   rule {rule:.3f}   CNN-MLP {T['F1'].loc[r, 'CNN-MLP']:.3f}   difference {gap[-1]:+.3f}")
    print(f"   mean difference over the 12 experiments: {np.mean(gap):+.3f}")


if __name__ == "__main__":
    main()
