#!/bin/bash
# Step 3 batch: as-evaluated probes at the other heights, removal tests, CPU-physics collision check.
cd $(dirname $0)
D=$HOME/Projects/k1_research/airc2027_renders/diagnosis/sim
HA=mesh_2,mesh_3,mesh_4,mesh_5,mesh_6,mesh_7,mesh_8,mesh_9,mesh_10,mesh_11
AR=mesh_4,mesh_5,mesh_6,mesh_7,mesh_8,mesh_9,mesh_10,mesh_11
HD=mesh_2,mesh_3
./run_gpu.sh step3_cz042 sim_step3.py --episode_idx=0 --cam_z=0.42 --skip_removal --out_dir $D/step3_cz042
./run_gpu.sh step3_cz025 sim_step3.py --episode_idx=0 --cam_z=0.25 --skip_removal --out_dir $D/step3_cz025
for cz in 057 042 025; do
  ./run_gpu.sh removal_headarms_cz$cz sim_step3.py --episode_idx=0 --cam_z=0.${cz:1} --skip_removal --hide_meshes $HA --work_dir /tmp/claude-1000/-home-boosterk1/f3269a68-3809-448e-983f-d8a64a1f1032/scratchpad/work --out_dir $D/removal_headarms_cz$cz
done
for cz in 057 042; do
  ./run_gpu.sh removal_armsonly_cz$cz sim_step3.py --episode_idx=0 --cam_z=0.${cz:1} --skip_removal --walk_steps 0 --hide_meshes $AR --work_dir /tmp/claude-1000/-home-boosterk1/f3269a68-3809-448e-983f-d8a64a1f1032/scratchpad/work --out_dir $D/removal_armsonly_cz$cz
  ./run_gpu.sh removal_headonly_cz$cz sim_step3.py --episode_idx=0 --cam_z=0.${cz:1} --skip_removal --walk_steps 0 --hide_meshes $HD --work_dir /tmp/claude-1000/-home-boosterk1/f3269a68-3809-448e-983f-d8a64a1f1032/scratchpad/work --out_dir $D/removal_headonly_cz$cz
done
./run_gpu.sh physicscpu_cz025 sim_step3.py --episode_idx=0 --cam_z=0.25 --skip_removal --physics_cpu --out_dir $D/physics_cpu_diag_cz025
echo BATCH_DONE
