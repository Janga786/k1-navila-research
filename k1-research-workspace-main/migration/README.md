# migration/ — clone the K1 / NaVILA stack to a new machine

Self-contained bundle to reproduce the working NaVILA + K1 + Isaac stack on a new
lab machine (target: **RTX 3090**). Start with **[`MIGRATION.md`](MIGRATION.md)**.

```
migration/
├── MIGRATION.md           full runbook (read this)
├── bootstrap.sh           staged reconstruction: prereqs|repos|policies|env-navila|weights|env-isaac|verify|all
├── capture.sh             how this bundle was generated on SOURCE (reference)
├── repos.manifest.tsv     every sub-repo: upstream, branch, pinned commit
├── patches/               your local diffs vs each public upstream (third-party code is NOT re-hosted)
├── untracked/             non-regenerable untracked dev files, per repo (tar.gz)
├── env/                   exact conda/pip recipes for all 7 envs + bashrc additions
└── policies/              trained K1 locomotion policies, split into <100MB parts (+ sha256)
```

Quick start on TARGET (after MIGRATION.md §1 prerequisites):

```bash
git clone https://github.com/Janga786/k1-research-workspace.git ~/Projects/k1_research
cd ~/Projects/k1_research
bash migration/bootstrap.sh all
```

**Generated:** 2026-06-04 from SOURCE (`media`, RTX 5090). Regenerate with `bash migration/capture.sh`.
