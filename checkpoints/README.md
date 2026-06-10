# Policy checkpoints (deliberately committed — small, final, deployment-bound)

- `model_11999.pt` — K1 NaVILA velocity policy, production winner 2026-06-10.
  Raw rsl_rl checkpoint (the benchmark's build_v3_actor loads model_state_dict).
  235-dim obs / 12-DoF action contract; trained at proven gains 100/2 & 50/1.
  Full provenance: run 2026-06-09_13-52-48, iter 11999; circumduction fixed
  (foot_lat 0.070 vs 8700's 0.105), full tracking gate PASS, MuJoCo rated-torque
  gate PASS.

Benchmark usage on the 32 GB box:
  bash scripts/bench_diag.sh 0 10 smoke_11999 $(pwd)/checkpoints/model_11999.pt \
    "--closed_loop --clean_render --bright --max_episode_s 120 --vlm_transform stretch"
