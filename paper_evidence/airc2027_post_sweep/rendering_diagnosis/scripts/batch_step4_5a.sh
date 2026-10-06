#!/bin/bash
# Step 4 (repeatability, fresh processes) and Step 5A (as-evaluated robot-camera images), all as evaluated.
cd $(dirname $0)
D=$HOME/Projects/k1_research/airc2027_renders/diagnosis/repeatability
I=$HOME/Projects/k1_research/airc2027_renders/diagnosis/step5a_raw
for rep in 1 2; do
  for cz in 025 057; do
    ./run_gpu.sh rep41_cz${cz}_p$rep sim_capture.py --protocol init50 --episode_idx=0 --cam_z=0.${cz:1} --out_dir $D/step4_1_idx0_cz${cz}_proc$rep
  done
done
for rep in 1 2 3 4; do
  ./run_gpu.sh rep42_idx21_p$rep sim_capture.py --protocol mainloop --episode_idx=21 --cam_z=0.25 --out_dir $D/step4_2_idx21_cz025_proc$rep
done
for cz in 007 025 042 057 077 097; do
  ./run_gpu.sh s5a_idx0_cz$cz sim_capture.py --protocol mainloop --main_steps 1 --episode_idx=0 --cam_z=0.${cz:1} --out_dir $I/idx0_cz$cz
done
./run_gpu.sh s5a_idx7_cz025 sim_capture.py --protocol mainloop --main_steps 1 --episode_idx=7 --cam_z=0.25 --out_dir $I/idx7_cz025
echo BATCH_DONE
