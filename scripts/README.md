# scripts/

This path used to be a symlink to `k1-research-workspace-main/scripts`, a vendored directory that is not
part of this repository, so in a fresh clone it pointed nowhere.

The script from that directory used by the sweep, `run_powered_benchmark.sh`, is in
[`paper_evidence/airc2027_code/workspace/scripts/`](../paper_evidence/airc2027_code/workspace/scripts/).
`arms_runner.sh` calls it as `~/Projects/k1_research/k1-research-workspace-main/scripts/run_powered_benchmark.sh`.
[`paper_evidence/airc2027_code/README.md`](../paper_evidence/airc2027_code/README.md) explains how to recreate that
layout and describes every file the sweep used.
