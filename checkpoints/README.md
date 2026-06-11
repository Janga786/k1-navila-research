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

- `model_14498.pt` (+ `_jit.pt` / `.onnx`) — feet_distance fine-tune of model_11999
  (run 2026-06-10_14-30-20, iters 11999->14498). FIXES FOOT-TO-FOOT CLIPPING:
  MuJoCo probe 0 contacts in all phases (was 15.8% of steps at vx=0.5), stand
  width 0.183 m. Circ 0.085 (still well under 8700's 0.105), W4 rated-clamp PASS,
  W5 chatter same class. Tracking: all primitives PASS except the historically
  benign ramp-limited 1.5s-dist (0.626 < 0.65; closed-loop compensates).
  RECOMMENDED policy for hardware deploy AND the headline benchmark run.
