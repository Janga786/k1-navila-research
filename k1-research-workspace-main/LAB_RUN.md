# Run the K1 VLN-CE-Isaac benchmark on the lab machine (with the trained policy)

Everything from the navigation-side work is on GitHub (both repos **private**):
- **`Janga786/NaVILA-Bench`** (`main`) — the full working vision-nav benchmark code (closed-loop
  exec, render denoise+brightness, diagnostics, endgame fixes, K1 v3 wrappers/config).
- **`Janga786/k1-research-workspace`** — the runner + aggregator (`scripts/`, `aggregate_k1_vision_results.py`).

Assumes the standard layout from the migration bootstrap: `~/Projects/k1_research/` (the workspace
clone) with `NaVILA-Bench/` inside it, the `navila`/`vlnce-isaac` conda envs, Isaac Sim, and the
matterport assets already present (the lab box has these — it trained the policy).

## 1. Pull the latest code
```bash
# workspace (runner + aggregator)
cd ~/Projects/k1_research && git pull

# NaVILA-Bench: pull the updated nav code from the fork into the existing checkout
# (keeps your local assets/, which are gitignored). One-time remote add, then fetch.
cd ~/Projects/k1_research/NaVILA-Bench
git remote add labfork https://github.com/Janga786/NaVILA-Bench.git 2>/dev/null || true
git fetch labfork main
git checkout labfork/main -- \
    scripts/navila_eval_v3.py scripts/nav_diag.py \
    isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/wrappers_v3.py
# (sanity) python -c "import ast; ast.parse(open('scripts/navila_eval_v3.py').read()); print('ok')"
```
> Fresh machine instead? `git clone https://github.com/Janga786/NaVILA-Bench.git`, then
> re-download assets: `huggingface-cli download Zhaojing/VLN-CE-Isaac --repo-type dataset
> --local-dir NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/` and the NaVILA weights
> (`huggingface-cli download a8cheng/navila-llama3-8b-8f --local-dir
> booster/NaVILA/checkpoints/navila-llama3-8b-8f`).

## 2. Run the powered benchmark with the trained policy
```bash
cd ~/Projects/k1_research
# point --checkpoint at the newly-trained velocity policy on the lab box:
bash scripts/run_powered_benchmark.sh prod_newpolicy /ABS/PATH/TO/new_policy/model_XXXX.pt
```
That's it. The runner:
- auto-starts the NaVILA VLM bridge (port 54321, env `navila`),
- runs all 1077 episodes in env `vlnce-isaac` with the production config
  **`--closed_loop --clean_render --bright --max_episode_s 120`**,
- is **resumable** — re-run the same command after any crash; it skips episodes already scored,
- prints the aggregate (NE/OS/SR/SPL over the full 1077, missing=failures) + the paper table at the end.

Results land in `NaVILA-Bench/eval_results/k1_matterport_vision_loco_prod_newpolicy/`.
Re-aggregate anytime:
```bash
python aggregate_k1_vision_results.py \
  --measurements-dir NaVILA-Bench/eval_results/k1_matterport_vision_loco_prod_newpolicy/measurements \
  --total-episodes 1077 --paper-table
```

## 3. Useful variants
```bash
# quick diagnostic slice (saves trajectory plots + VLM frames per episode, ~20 eps):
bash scripts/bench_diag.sh 0 20 diag_newpolicy /ABS/PATH/new_policy.pt "--clean_render --bright --closed_loop --max_episode_s 120"

# OS->SR CEILING (ground-truth stop; NOT a fair benchmark number, just the ceiling):
bash scripts/run_powered_benchmark.sh ceiling_newpolicy /ABS/PATH/new_policy.pt 0 1077 "--proximity_stop 3.0"
```

## What to expect / where the wins are
- The nav side is fixed where it legitimately can be; with a policy that **tracks velocity accurately**
  (reaches the goal precisely), expect OS to rise and — critically — the VLM to emit **stop** on the
  clean, bright, in-distribution at-goal view, converting OS→SR. The proven SR ceiling on the current
  policy was 25% (=OS); a well-tracking policy should lift both OS and the stop-conversion toward the
  paper's Go2/H1 vision range (45–50%).
- Flags `--max_chunk_m` / `--stop_assist` are OFF by default (a wash / a regression respectively on the
  old policy); revisit `--max_chunk_m 0.5` once you confirm the new policy tracks, to bound any overshoot.

Full root-cause + A/B history: see the analysis in the workspace and `migration/` bundle.
