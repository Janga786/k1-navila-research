#!/bin/bash
# Step 5C corrected renders (same views as 5B), studio render, and three diagnostic follow-ups.
cd $(dirname $0)
R=$HOME/Projects/k1_research/airc2027_renders
TP=$R/diagnosis/trunk_pose_idx0_after_reset.json
GPU_TIMEOUT=900 ./run_gpu.sh pt_corrected sim_render_pt.py --model corrected --episode_idx=0 --trunk_pose_json $TP --views_json $R/images/as_evaluated/third_person/views.json --out_dir $R/images/corrected
GPU_TIMEOUT=900 ./run_gpu.sh pt_studio sim_render_pt.py --model studio --episode_idx=0 --trunk_pose_json $TP --out_dir $R/images/corrected/studio
# causal test: give mesh_1..mesh_11 the material binding mesh_0 has (copy of the converted USD, before load)
./run_gpu.sh matbind_cz057 sim_step3.py --episode_idx=0 --cam_z=0.57 --skip_removal --bind_material_meshes mesh_1,mesh_2,mesh_3,mesh_4,mesh_5,mesh_6,mesh_7,mesh_8,mesh_9,mesh_10,mesh_11 --work_dir /tmp/claude-1000/-home-boosterk1/f3269a68-3809-448e-983f-d8a64a1f1032/scratchpad/work --out_dir $R/diagnosis/sim/causal_material_binding_cz057
# the glyph-like shapes at 0.42: hide only the K1 logo mesh
./run_gpu.sh removal_logo_cz042 sim_step3.py --episode_idx=0 --cam_z=0.42 --skip_removal --walk_steps 0 --hide_meshes mesh_1 --work_dir /tmp/claude-1000/-home-boosterk1/f3269a68-3809-448e-983f-d8a64a1f1032/scratchpad/work --out_dir $R/diagnosis/sim/removal_logoonly_cz042
# per-episode pattern at 0.42 (h095): two episodes gray at video frame 1, one never gray
for i in 75 81 72; do
  ./run_gpu.sh h095pattern_idx$i sim_capture.py --protocol mainloop --main_steps 1 --episode_idx=$i --cam_z=0.42 --out_dir $R/diagnosis/h095_pattern/idx${i}_cz042
done
echo BATCH_DONE
