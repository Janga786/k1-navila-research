"""Where divergence starts (Section V-A).

Two pre-sweep pairs of runs evaluated the same six episodes twice each with the production
settings plus --diag, which logs every model query: the step count, the robot's root position and
heading at the query, and the model's reply. Pair 1 (exta/extb, 2026-07-23) ran under driver
580.159.03, pair 2 (smoke173a/b, 2026-07-24) under 580.173.02, the sweep driver.
Source: data/probe_ticks/<run>/ep<i>/ticks.jsonl, copied from the private archive
(NaVILA-Complete-Archive/12_airc2027_materials/probes/*.tar.gz); the frames were not saved.

For each episode pair we find the first query whose reply differs and check whether the logged
pose and step count were still bit-identical there (JSON floats round-trip exactly).
Writes tables/probe_divergence.json. Deterministic.
"""
import json, os
from common import HERE, TABLES

D = os.path.join(HERE, "data", "probe_ticks")
PAIRS = [("exta", "extb", "580.159.03"), ("smoke173a", "smoke173b", "580.173.02")]


def load(run, ep):
    with open(os.path.join(D, run, ep, "ticks.jsonl")) as f:
        return [json.loads(line) for line in f]


rows = []
for a_run, b_run, driver in PAIRS:
    for ep in sorted(os.listdir(os.path.join(D, a_run))):
        a, b = load(a_run, ep), load(b_run, ep)
        n = min(len(a), len(b))
        first_reply = next((k for k in range(n) if a[k]["vlm_raw"] != b[k]["vlm_raw"]), None)
        first_pose = next((k for k in range(n)
                           if (a[k]["robot_xy"], a[k]["robot_yaw"], a[k]["num_steps"]) !=
                              (b[k]["robot_xy"], b[k]["robot_yaw"], b[k]["num_steps"])), None)
        identical = first_reply is None and first_pose is None and len(a) == len(b)
        row = dict(pair=f"{a_run}/{b_run}", driver=driver, episode=ep, ticks=[len(a), len(b)],
                   identical=identical, first_reply_diff=first_reply, first_pose_or_step_diff=first_pose)
        if first_reply is not None:
            k = first_reply
            row.update(step_at_first_reply_diff=a[k]["num_steps"],
                       pose_identical_at_first_reply_diff=(a[k]["robot_xy"], a[k]["robot_yaw"], a[k]["num_steps"]) ==
                                                         (b[k]["robot_xy"], b[k]["robot_yaw"], b[k]["num_steps"]),
                       replies=[a[k]["vlm_raw"], b[k]["vlm_raw"]])
        rows.append(row)

diverged = [r for r in rows if not r["identical"]]
summary = dict(pairs=len(rows), diverged=len(diverged),
               reply_first_at_identical_pose=sum(1 for r in diverged if r.get("pose_identical_at_first_reply_diff")),
               pose_diverged_before_reply=sum(1 for r in diverged if r["first_pose_or_step_diff"] is not None and
                                              (r["first_reply_diff"] is None or r["first_pose_or_step_diff"] < r["first_reply_diff"])))
out = dict(summary=summary, pairs=rows)
with open(os.path.join(TABLES, "probe_divergence.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    print(r["pair"], r["episode"], "identical" if r["identical"] else
          f"reply differs first at query {r['first_reply_diff']} (step {r['step_at_first_reply_diff']}), "
          f"pose identical there: {r['pose_identical_at_first_reply_diff']}; first pose/step difference at query {r['first_pose_or_step_diff']}")
