"""Remaining manuscript numbers not written by paper_numbers.py / paper_numbers_v2.py.

 * J's bit-identical pairs with D1-D3: count longer than 2,000 steps and count ending at the
   step cap (sweep run's ended_at_step; interrupted records excluded, as in Fig. 1a);
 * handler-written interruption records after step 5,500 and at step 6,001;
 * interrupted sweep records that are bit-identical to J's record for the same episode;
 * share of sweep episodes whose navigation reached the step cap (ended_at_step >= 6,001,
   any termination label) and share labeled step_cap;
 * SR on bit-identical versus other D1-D2 episodes.
Writes tables/extras.json. Deterministic.
"""
import json, os
import numpy as np
from common import load_runs, pinned_sets, wide, TABLES


def main():
    df = load_runs(); S300, S200 = pinned_sets(df)
    E = {}
    # J identical pairs with D1-D3 (Fig. 1a orange set)
    long_, cap_ = 0, 0
    for r, idx in [("A", S300), ("B", S300), ("h078", S200)]:
        S = wide(df, "sig", ["base", r], idx)
        L = wide(df, "ended_at_step", [r], idx)[r].values
        T = wide(df, "timeout", [r], idx)[r].values
        ident = np.array([x == y for x, y in zip(S["base"], S[r])])
        keep = ident & (T == 0) & (L >= 0)
        long_ += int((keep & (L > 2000)).sum()); cap_ += int((keep & (L >= 6001)).sum())
    E["J_identical_longer_than_2000"] = long_
    E["J_identical_at_cap"] = cap_
    # handler-written interruption timing
    sw = df[df.run != "base"]
    to = sw[(sw.timeout == 1) & (sw.ended_at_step >= 0)]
    E["handler_written"] = int(len(to))
    E["handler_after_5500"] = int((to.ended_at_step > 5500).sum())
    E["handler_at_6001"] = int((to.ended_at_step >= 6001).sum())
    E["handler_min_step"] = float(to.ended_at_step.min())
    # interrupted sweep records identical to J
    J = df[df.run == "base"].set_index("record_idx")["sig"]
    hits = []
    for _, row in sw[sw.timeout == 1].iterrows():
        if row.record_idx in J.index and J.loc[row.record_idx] == row.sig:
            hits.append(dict(run=row.run, record_idx=int(row.record_idx), step=float(row.ended_at_step)))
    E["interrupted_identical_to_J"] = hits
    # step-cap shares
    E["cap_share_pct"] = {r: dict(labeled=100 * float((d.term_reason == "step_cap").mean()),
                                  reached=100 * float((d.ended_at_step >= 6001).mean()))
                          for r, d in sw.groupby("run")}
    # SR on identical versus other D1-D2 episodes
    S = wide(df, "sig", ["A", "B"], S300); Y = wide(df, "success", ["A", "B"], S300).values.astype(float)
    ident = np.array([x == y for x, y in zip(S["A"], S["B"])])
    E["AB_SR_identical_pct"] = 100 * float(Y[ident].mean())
    E["AB_SR_other_pct"] = 100 * float(Y[~ident].mean())
    with open(os.path.join(TABLES, "extras.json"), "w") as f:
        json.dump(E, f, indent=1)
    print(json.dumps(E, indent=1))


if __name__ == "__main__":
    main()
