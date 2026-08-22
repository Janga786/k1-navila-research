# KNOWN GAPS — what this evidence base does NOT support

Compiled 2026-08-22, after the sweep completed, by auditing every paper-facing claim in this
repository against the primary data on disk.

The purpose of this file is adversarial: it lists the places where a reviewer can push and the
repository would not hold. Items are ordered by how much damage they do if found by someone
else first. Anything fixed is recorded as fixed, with the commit that fixed it.

---

## RESOLVED during the 2026-08-22 audit

**R1. The analysis code mis-selected the baseline episode set.**
`analyze_sweep.py` subset the 1077-episode baseline by `record_idx < 300`. Benchmark episode
ids are sparse, so that selected 195 of the arms' 300 episodes and scored the missing 105 as
failures — baseline SR 10.0% instead of the true 17.7%. It produced two *significant*
driver-change effects that do not exist. Fixed: the pinned set is now derived from the arm
receipts themselves and verified in `ANALYSIS.md` §0; the corrected baseline reproduces the
pre-registered anchor (17.7 / 31.7 / NE 7.22) exactly. Corrected result: the driver-change
contrasts are **null**, i.e. the sweep reproduced the baseline.

**R2. The code that produced the sweep was never committed.** The `--cam_z` camera override
(the entire height sweep), the SIGTERM wall-timeout handler, and the `-1.0` sentinel exclusion
in the aggregator ran uncommitted for the whole sweep, while `AWAY_STATUS.md` claimed "all
committed + pushed." No pushed commit could reproduce any arm. Fixed in `25f7160`.

**R3. The analysis plan changed without a written amendment.** The pre-registration names
McNemar as the primary test; the sweep used unpaired two-proportion tests. The substitution
was correct (forced by the pre-registration's own determinism gate) but existed only in
`HANDOFF.md`, and `ANALYSIS.md` cited an "AMENDMENT section" that did not exist. Fixed:
Amendments 1 and 2 appended to `PRE_REGISTRATION.md`.

**R4. The pre-registered reproducibility gate was never adjudicated in writing.** The
pre-registration says STOP if stretch@300 fails to reproduce the anchor; the runner proceeded
through nine further arms with no recorded ruling. Fixed: Amendment 2 adjudicates it — it
reproduced (all contrasts null), with the explicit caveat that "no detectable difference at
this power" is not equivalence.

**R5. Environment and model provenance were unrecorded.** No pip freeze, no checkpoint hash,
no dataset hash, and the raw runner log lived only in volatile `/tmp` under a gitignored
extension. Fixed: `receipts/provenance/` and the dataset SHA-256 in `ROBUSTNESS.md` §D.

**R6. A locally modified simulator config was unversioned anywhere.** IsaacLab's PhysX GPU
buffer capacities are reduced to fit the 24 GB card. Fixed: captured as
`receipts/provenance/isaaclab_simulation_cfg.patch` and documented in `ENVIRONMENT.md`, with
the open question (below, G4) stated rather than buried.

---

## OPEN — cannot be closed without re-running compute

**G1. The two root-cause findings have no receipts. This is the most serious open gap.**
`PRE_REGISTRATION.md` asserts:
- *5b:* "50 real VLM queries across 10 episodes, each fed twice, token-diffed, 50/50 identical."
- *5a:* "same pose in two processes -> frames differ, max |diff| 21/255, mean 0.30, 25.68% of
  channels nonzero."

Together these carry the project's most novel claim — *chaotic divergence originates in the
simulator, not the VLM*. Neither has a script, a log, or an output artifact in this repo. The
commit that added them (`fc31cb8`) touched only the markdown. Worse, the one related receipt,
`receipts/prelaunch_suite_report.txt`, records the 5a camera dump as having **failed to produce
PNGs**, and shows only a *single* query pair for 5b, not fifty.

The numbers may well be real — they are too specific to be invented — but they are currently
unverifiable, and the claim is load-bearing. **Do not put 5a/5b numbers in the paper until the
probes are re-run and their outputs committed.** Estimated cost ~1 GPU-hour. Until then, the
defensible substitute already exists and is far stronger: `ROBUSTNESS.md` §A quantifies
run-to-run instability over 300 paired episodes (11.0% discordant, kappa 0.58) — but note it
demonstrates *that* the pipeline is non-deterministic, not *where* the nondeterminism
originates. The sim-vs-VLM attribution rests entirely on G1.

**G2. The wall-timeout guarantee was never successfully acceptance-tested.**
`receipts/prelaunch_suite_report.txt` TEST 1 records `wall_timeout marker in log: 0`,
`JSON produced: NONE` — the in-eval SIGTERM handler did not fire under test. The guarantee that
every episode yields a record was in practice delivered by a *second*, runner-side stub writer
in `run_powered_benchmark.sh`, which was never acceptance-tested at all. Two writers with
slightly different schemas both produced records in the shipped dataset (see `ROBUSTNESS.md`
§E: 9 records carry `max_episode_steps = -1`).

Mitigating: the delivered data is clean. All 2,400 episodes produced records, 0 missing across
all ten arms, and all 151 wall-timeout records satisfy the pre-registered spec (`success=0`,
`spl=0`, with 34 legitimately retaining `oracle_success=1`). The mechanism is undocumented and
untested; the outcome is verified. Say so plainly rather than claiming a tested guarantee.

**G3. Height-sweep manipulation checking is weak.** The only receipt confirming that `--cam_z`
actually moved the camera is the `[cam-override]` print line plus a FOV/resolution invariance
check. That receipt's own Trunk-z diagnostic reports a mean of **-1.887 m** against a nominal
0.53 m, which cannot be right in the frame the paper will quote. No rendered frames were
retained at any height. A reviewer asking "show me the camera was where you say" cannot be
answered from this repository. Cheap fix: dump one frame per height from a fixed pose and
commit the six images.

**G4. Whether the reduced PhysX buffers cause the nondeterminism is untested.** Buffer
capacities are allocation knobs and should not change dynamics — unless one overflows, in
which case PhysX drops contacts silently. This was never tested, and it is a plausible
mechanism for the sim-side chaos in G1. It is a constant across the baseline and every arm
(verified by mtime), so it is not a differential confound between arms, but it may be part of
the causal story the paper tells about nondeterminism.

**G5. No qualitative evidence exists for any sweep episode.** 2,400 episodes ran; no video and
no sample frames were retained. `receipts/media_index.md` predates the sweep. Nothing supports
a behavioural figure or a failure-mode taxonomy for the sweep itself.

---

## OPEN — documentation-only, closable without compute

**G6. `FINAL_RESULTS_full1077.md` and `SLIDE_FACTS.md` carry a claim the sweep refutes.**
Both state, from an n=10 pilot, that crop "reached but never stopped" and conclude peripheral
FOV is required for arrival recognition. At n=300 the direction inverts: crop converts 69.4% of
reaches into stops vs stretch's 51.1% (`ROBUSTNESS.md` §C). A correction notice has been added
to the head of both files; the bodies are left intact rather than silently rewritten, so the
original claim and its refutation are both on the record. The underlying text still needs a
proper rewrite before reuse.

**G7. `EVIDENCE.md` lists 13 required edits to `FINAL_RESULTS_full1077.md`; none were applied.**
Including a garbled sentence ("scores SR 0/10 ... scores 30/10", line 53) and the wording
`EVIDENCE.md` claim 12 explicitly forbids.

**G8. `HANDOFF.md` cites a "single-pass-equivalent 16.7%" baseline that no file derives.**
If that, rather than 17.7%, was the intended reproducibility target, it changes how the gate in
R4 reads. It needs a derivation or a retraction.

**G9. Multiplicity structure is post-hoc.** `ANALYSIS.md` applies Holm within contrast families,
but the pre-registration never names families, never specifies a correction, and never
designates SR or OS as the primary endpoint. With 22 reported contrasts this is not
recoverable after the fact. The honest framing is that all inference is exploratory.

**G10. Methods facts that live only in code.** The `record_idx = episode_id - 1` mapping and the
sparse-index consequence (now in `episode_index.csv`); the definition of NE as *geodesic*
distance along ground-truth waypoints via KD-tree, not Euclidean; the termination taxonomy,
where **`sim_done` (13-30 episodes per arm) is defined nowhere**; and the fact that
`max_episode_s 120` is *simulation* seconds while `EP_TIMEOUT=900` is *wall-clock* seconds.

**G11. Compute cost is unrecorded.** ~2,400 episodes at ~600 s each is roughly **400 GPU-hours**
for the sweep. `receipts/run_ledger.md`'s "270 GPU-h" covers training only.

**G12. `SWEEP_STATUS.md` records only SR and OS** — no NE, ONE, SPL or path length for any arm.
They are recomputed in `ANALYSIS.md` §1; SPL in particular varies more across arms (7.0-11.2%)
than SR does.

**G13. `h078` is an unacknowledged third same-config replicate.** It is `--cam_z=0.25`, the
config default — identical to stretchA/stretchB apart from episode count. `ANALYSIS.md` §3.3
uses it as the fixed control for the h095 peak without noting the control is itself one draw
from a distribution known to span a couple of points.

**G14. Stale operational docs contradict each other.** `RUN_STATUS.md` announces "REMOTE ACCESS:
NOT REACHABLE OFF-NETWORK — Tailscale not installed"; `AWAY_STATUS.md` and `HANDOFF.md` document
a working Tailscale setup. `SLIDE_FACTS.md` still lists the abandoned 100-episode A/B as "IN
PROGRESS since 07-22."

**G15. The "~24x improvement" headline rests on an ungeneratable baseline.** The v1 SR 0.76%
figure is marked in `EVIDENCE.md` as not on disk and not regenerable. Defensible as "recorded
prior-stack baseline"; not as a measured comparison.

---

## The one-line summary a reviewer should get

The sweep's *primary* claims — ten arms, 2,400 episodes, all transform and height contrasts
null, reproduction of the baseline across a driver change, and aggregate-stable /
episode-unstable non-determinism — are fully backed by committed per-episode data and
re-runnable analysis code. The *secondary* mechanistic claims — where nondeterminism
originates (G1), and peripheral FOV driving arrival recognition (G6) — are not: one has no
receipts, and the other is refuted by the sweep itself.
