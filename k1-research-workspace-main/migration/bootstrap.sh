#!/usr/bin/env bash
# bootstrap.sh — reconstruct the K1 / NaVILA workspace on a new lab machine.
#
# Run AFTER cloning the workspace repo, e.g.:
#   git clone https://github.com/Janga786/k1-research-workspace.git ~/Projects/k1_research
#   cd ~/Projects/k1_research && bash migration/bootstrap.sh <stage>
#
# Stages (run in order; each is safe to re-run):
#   prereqs    - check tools (conda, git, gh, nvidia, ros) — does NOT install anything
#   repos      - clone every sub-repo at its pinned commit, apply local patches, restore untracked files
#   policies   - reassemble + extract the trained locomotion policies (booster_train/logs)
#   env-navila - build the 'navila' conda env (NaVILA inference + K1 deploy)
#   weights    - download NaVILA model weights (16G) + (optionally) NaVILA-Bench scene data
#   env-isaac  - build the Isaac Sim / IsaacLab / training envs (HEAVY, ~80GB; full-clone only)
#   verify     - sanity-check that NaVILA imports and the policy bundle is intact
#   all        - prereqs -> repos -> policies -> env-navila -> weights  (skips heavy isaac stage)
#
# TARGET GPU: RTX 3090 (Ampere sm_86). torch 2.7.1+cu128 runs on it unchanged.
# Note: 24GB VRAM vs the source 32GB — the 8B model fits for inference; watch long contexts.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIG="$ROOT/migration"
MANIFEST="$MIG/repos.manifest.tsv"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
say()  { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

stage_prereqs() {
  say "Checking prerequisites (no installs performed)"
  for t in git conda python3 huggingface-cli; do
    command -v "$t" >/dev/null && echo "  ok: $t ($(command -v $t))" || warn "MISSING: $t"
  done
  command -v gh >/dev/null && gh auth status 2>&1 | sed 's/^/  gh: /' || warn "gh CLI not authed — needed to clone the private k1-vlm-navigation repo (or use a PAT)"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  gpu: /'
  else warn "nvidia-smi missing — install the NVIDIA driver (>=535 for cu128 on the 3090)"; fi
  [ -d /opt/ros/jazzy ] && echo "  ok: ROS 2 Jazzy at /opt/ros/jazzy" || \
    warn "ROS 2 Jazzy NOT found. The navila env contains ROS (rclpy/ament). Install it FIRST: see MIGRATION.md §1"
  echo "  apt deps needed: build-essential cmake ninja-build ffmpeg libgl1 libegl1 libglib2.0-0 (see MIGRATION.md §1)"
}

stage_repos() {
  [ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
  say "Reconstructing sub-repos from $MANIFEST"
  # columns: name path upstream branch commit dirty_files
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r name rel upstream branch commit dirty; do
    [ "$rel" = "." ] && continue                       # the workspace repo IS the current checkout
    dest="$ROOT/$rel"
    if [ "$name" = "k1-vlm-navigation" ]; then
      say "clone $name (private) -> $rel  [branch $branch]"
      [ -d "$dest/.git" ] || git clone "$upstream" "$dest" || { warn "clone $name failed (auth? gh/PAT). Skipping."; continue; }
      git -C "$dest" fetch origin "$branch" 2>/dev/null && git -C "$dest" checkout "$branch" 2>/dev/null
      continue
    fi
    say "clone $name -> $rel  [pin $commit on $branch]"
    if [ ! -d "$dest/.git" ]; then
      git clone "$upstream" "$dest" || { warn "clone $name failed. Skipping."; continue; }
    fi
    git -C "$dest" fetch --all --tags 2>/dev/null
    if git -C "$dest" checkout "$commit" 2>/dev/null; then
      echo "  checked out pinned commit $commit"
    else
      warn "  pinned commit $commit not fetchable; falling back to branch '$branch' HEAD"
      git -C "$dest" checkout "$branch" 2>/dev/null || warn "  could not checkout $branch either"
    fi
    # apply local modifications
    if [ -f "$MIG/patches/$name.patch" ]; then
      if git -C "$dest" apply --check "$MIG/patches/$name.patch" 2>/dev/null; then
        git -C "$dest" apply "$MIG/patches/$name.patch" && echo "  applied patch: $name.patch"
      else
        warn "  patch $name.patch did not apply cleanly — apply by hand (git apply --3way migration/patches/$name.patch)"
      fi
    fi
    # restore untracked, non-regenerable files
    if [ -f "$MIG/untracked/$name.untracked.tgz" ]; then
      tar -C "$dest" -xzf "$MIG/untracked/$name.untracked.tgz" && echo "  restored untracked files for $name"
    fi
  done
  say "repos done. Review with: cd <each repo>; git status"
}

stage_policies() {
  say "Reassembling trained locomotion policies"
  local parts="$MIG/policies/parts" tgz="$MIG/policies/k1_locomotion_policies.tgz"
  ls "$parts"/k1_locomotion_policies.tgz.part*.bin >/dev/null 2>&1 || die "policy parts not found in $parts"
  cat "$parts"/k1_locomotion_policies.tgz.part*.bin > "$tgz"
  if [ -f "$MIG/policies/k1_locomotion_policies.tgz.sha256" ]; then
    echo "$(cat "$MIG/policies/k1_locomotion_policies.tgz.sha256")  $tgz" | sha256sum -c - || die "checksum mismatch — re-pull the parts"
  fi
  mkdir -p "$ROOT/booster/booster_train"
  tar -C "$ROOT/booster/booster_train" -xzf "$tgz" && echo "  extracted policies into booster/booster_train/logs/"
  rm -f "$tgz"
}

stage_env_navila() {
  say "Building the 'navila' conda env (inference + deploy)"
  [ -f "$CONDA_SH" ] || die "miniconda not at ~/miniconda3 — install it first (see MIGRATION.md §1)"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda env list | grep -q '/envs/navila$' || conda create -y -n navila python=3.10.20
  conda activate navila
  say "  torch first (cu128 wheel — runs on the 3090 sm_86)"
  pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128
  say "  remaining pip deps (ROS/editable/local/torch lines filtered — handled separately)"
  local req="$MIG/env/navila.requirements.txt" filt; filt="$(mktemp)"
  grep -vE '(\+cu128|^-e |@ file://|@ git\+|^(ament|rclpy|rosidl|rcl-|rpyutils|rosgraph|rosdistro|ros2|launch.?ros|tf2|.*-msgs==|sensor-msgs|std-msgs|geometry-msgs|action-msgs|builtin-interfaces|domain-coordinator|rosbag2|rmw-))' "$req" > "$filt"
  pip install -r "$filt" || warn "  some pip deps failed — inspect $filt; ROS pkgs are expected to come from /opt/ros/jazzy"
  rm -f "$filt"
  say "  git + editable installs (from CORRECT target paths — fixes the source machine's broken vila path)"
  pip install "git+https://github.com/bfshi/scaling_on_scales.git@9c008a37540e761f53574b488979db6e49a64312" || warn "s2wrapper install failed"
  [ -d "$ROOT/booster/booster_robotics_sdk" ] && pip install -e "$ROOT/booster/booster_robotics_sdk" || warn "booster_robotics_sdk not present — run 'repos' first"
  [ -d "$ROOT/booster/NaVILA" ] && pip install -e "$ROOT/booster/NaVILA" || warn "NaVILA not present — run 'repos' first"
  say "  navila env built. ROS reminder: ensure /opt/ros/jazzy is sourced in ~/.bashrc"
}

stage_weights() {
  say "Downloading NaVILA model weights (~16GB) from Hugging Face"
  command -v huggingface-cli >/dev/null || die "huggingface-cli missing — pip install huggingface_hub (in the navila env)"
  huggingface-cli download a8cheng/navila-llama3-8b-8f \
    --local-dir "$ROOT/booster/NaVILA/checkpoints/navila-llama3-8b-8f" \
    && echo "  weights -> booster/NaVILA/checkpoints/navila-llama3-8b-8f"
  say "  (full-clone) NaVILA-Bench scene data (~5.2GB) — uncomment to fetch:"
  echo "  # huggingface-cli download Zhaojing/VLN-CE-Isaac --repo-type dataset --local-dir \\"
  echo "  #     $ROOT/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/"
}

stage_env_isaac() {
  say "Building Isaac Sim / IsaacLab / training envs (HEAVY ~80GB) — full clone only"
  [ -f "$CONDA_SH" ] || die "miniconda missing"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  cat <<EOF
  This stage is intentionally guided, not blind — Isaac Sim is a multi-GB pip install
  and IsaacLab needs its own setup. Recommended order on the TARGET:

  1) Isaac Sim 5.1.0 (pip):
       conda create -y -n isaac_sim python=3.11 && conda activate isaac_sim
       pip install --upgrade pip
       pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
  2) IsaacLab v2.3.0:
       git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
       cd ~/IsaacLab && git checkout v2.3.0 && ./isaaclab.sh --install
  3) Recreate the project envs from the committed recipes (reference pins):
       migration/env/{isaacLab,isaacLab_311,vlnce-isaac,booster,vla}.environment.yml
       e.g.  conda env create -n vlnce-isaac -f migration/env/vlnce-isaac.environment.yml
  4) Re-run NaVILA-Bench / training using the cloned repos (repos stage) + the policies (policies stage).

  RTX 3090 note: 24GB VRAM is fine for K1 locomotion training and for NaVILA inference.
EOF
}

stage_verify() {
  say "Verifying"
  [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate navila 2>/dev/null
  python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import torch
    print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available(),
          "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
except Exception as e:
    print("torch check FAILED:", e)
for m in ("vila", "llava", "transformers"):
    try:
        __import__(m); print(m, "import OK")
    except Exception as e:
        print(m, "import FAILED:", e)
PY
  local nv="$ROOT/booster/NaVILA/checkpoints/navila-llama3-8b-8f"
  [ -d "$nv" ] && echo "  weights present ($(du -sh "$nv" 2>/dev/null | cut -f1))" || warn "  weights missing — run 'weights' stage"
  [ -d "$ROOT/booster/booster_train/logs/rsl_rl" ] && echo "  policies present" || warn "  policies missing — run 'policies' stage"
  echo "  end-to-end: cd $ROOT/experiments/navila && python test_navila.py"
}

main() {
  local stage="${1:-all}"
  case "$stage" in
    prereqs)    stage_prereqs ;;
    repos)      stage_repos ;;
    policies)   stage_policies ;;
    env-navila) stage_env_navila ;;
    weights)    stage_weights ;;
    env-isaac)  stage_env_isaac ;;
    verify)     stage_verify ;;
    all)        stage_prereqs; stage_repos; stage_policies; stage_env_navila; stage_weights; stage_verify
                say "Core (inference+deploy) done. For the full Isaac training stack: bash migration/bootstrap.sh env-isaac" ;;
    *)          die "unknown stage '$stage'. One of: prereqs repos policies env-navila weights env-isaac verify all" ;;
  esac
}
main "$@"