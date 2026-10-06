#!/bin/bash
# Step 6: start the model server exactly as run_powered_benchmark.sh does (8-bit), send the payload 20x, then restart
# the server twice and send it again each time. Only this step runs the model server.
set -u
R=$HOME/Projects/k1_research/airc2027_renders
NB=$HOME/Projects/k1_research/NaVILA-Bench
OUT=$R/diagnosis/model_repeat/replies.jsonl
: > $OUT
source ~/miniconda3/etc/profile.d/conda.sh
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy"; nvidia-smi; exit 9; fi
if ss -ltn | grep -q ":54321 "; then echo "port 54321 already in use"; exit 8; fi
for s in 1 2 3; do
  LOG=$R/logs/step6_server_session$s.log
  t0=$(date +%s)
  # --- as in run_powered_benchmark.sh (VLM_BRIDGE_EXTRA=--load_8bit, as in arms_runner.sh) ---
  setsid bash -c "
      cd $NB; source ~/miniconda3/etc/profile.d/conda.sh; conda activate navila
      exec python scripts/vlm_server_bridge.py \
          --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
          --port 54321 --load_8bit > $LOG 2>&1
  " < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 90); do grep -q "listening on" $LOG 2>/dev/null && break; sleep 5; done
  grep -q "listening on" $LOG || { echo "server not ready (session $s)"; tail -20 $LOG; exit 2; }
  SPID=$(pgrep -f "[v]lm_server_bridge.py --model_path" | head -1)
  echo "session $s: server pid $SPID ready after $(( $(date +%s) - t0 )) s"
  python3 $R/scripts/step6_client.py session$s 20 2 $OUT
  kill $SPID; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -9 $SPID 2>/dev/null
  t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%d\t%s\n" "step6_session$s" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" 0 "vlm_server_bridge.py --load_8bit + step6_client.py session$s 20 2" >> $R/logs/gpu_time.tsv
  sleep 3
done
echo STEP6_DONE
