# K1 / NaVILA → New Lab Machine (RTX 3090) — Migration Runbook

This reproduces the **complete working NaVILA + K1 stack** (vision-language navigation
inference, K1 robot deployment, and the Isaac Sim training/benchmark stack) on a new
lab machine, via **GitHub for all code + re-download of public bulk + a small bundled
set of irreplaceable artifacts**.

- **SOURCE** = the desktop "media" (RTX 5090, Ubuntu, `/home/janga`).
- **TARGET** = the new lab machine (**RTX 3090 / Ampere sm_86**, same Ubuntu).
- Everything you need lives in **two private GitHub repos** + public re-downloads:
  - `Janga786/k1-research-workspace` (this repo — docs, scripts, **`migration/` bundle**)
  - `Janga786/k1-vlm-navigation` (the NaVILA + K1 nav/voice/deploy stack — now **private**)

> **RTX 3090 vs 5090.** The pinned `torch==2.7.1+cu128` runs unchanged on Ampere (sm_86) —
> no rebuild needed, and the 3090 needs only a driver ≥ 535 (much easier than the 5090's ≥ 570).
> The one watch-item is VRAM: **24 GB vs 32 GB**. The NaVILA `llama3-8b-8f` model (~16 GB fp16)
> fits for inference, but keep an eye on very long video contexts. K1 locomotion training fits 24 GB fine.

> ⚠️ **Rotate the Groq API key.** It was hard-coded in `launch_demo.sh` and `~/.bashrc` on SOURCE
> (`gsk_3sig…`). It is now in a *private* repo, but it has been sitting in plaintext — rotate it at
> console.groq.com and set the new value via `export GROQ_API_KEY=…` on TARGET. Same for the SSH
> passwords in the launch scripts — switch to SSH keys.

---

## TL;DR (fast path)

```bash
# On TARGET, after the §1 prerequisites are installed:
git clone https://github.com/Janga786/k1-research-workspace.git ~/Projects/k1_research
cd ~/Projects/k1_research
bash migration/bootstrap.sh all          # prereqs -> repos -> policies -> navila env -> weights -> verify
bash migration/bootstrap.sh env-isaac    # then the heavy Isaac training stack (guided)
```

`bootstrap.sh` is staged and re-runnable. Read §1 first — it does **not** install system packages for you.

---

## 1. TARGET prerequisites (install these first)

```bash
# Base build/runtime libs
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build gcc g++ python3-dev \
    ffmpeg libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libosmesa6 curl wget git rsync

# NVIDIA driver for the RTX 3090 (>=535 supports the cu128 runtime). Match or exceed SOURCE.
sudo ubuntu-drivers autoinstall    # or: sudo apt-get install -y nvidia-driver-550
nvidia-smi                         # confirm the 3090 is visible

# ROS 2 Jazzy — REQUIRED: the navila env contains rclpy/ament packages and the robot
# bridge sources /opt/ros/jazzy. Install desktop before creating the conda env.
# https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
sudo apt-get install -y ros-jazzy-desktop
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

# Miniconda (if absent)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && source ~/.bashrc

# GitHub auth (needed to clone the private nav repo)
gh auth login        # or configure a PAT with 'repo' scope

# No system CUDA toolkit / nvcc is required for inference — the cu128 wheels bundle their own
# CUDA runtime. Only install the CUDA toolkit if you later compile CUDA ops (flash-attn etc.).
```

Then apply the SOURCE shell additions: see `migration/env/bashrc_additions.sh`
(ROS source line, conda init, and a placeholder for your **rotated** Groq key).

---

## 2. Reconstruct the code (automated)

```bash
git clone https://github.com/Janga786/k1-research-workspace.git ~/Projects/k1_research
cd ~/Projects/k1_research
bash migration/bootstrap.sh repos
```

This clones every sub-repo at its **pinned commit**, re-applies your **local patches**, and
restores **untracked dev files**. The map (`migration/repos.manifest.tsv`):

| Repo | Upstream | Branch | Pinned commit | Local changes captured |
|---|---|---|---|---|
| `booster/NaVILA` | AnjieCheng/NaVILA | main | `76b98f2` | patch (flash_attn + deepspeed.comm graceful fallback) |
| `NaVILA-Bench` | yang-zj1026/NaVILA-Bench | vlnce-isaac-benchmark | `eb56f0d` | patch (3 files) + untracked `wrappers_v3.py`, `navila_eval_v3.py` |
| `legged-loco` | yang-zj1026/legged-loco | k1-vision | `c8fc1a1` | patch (k1 vision cfg) |
| `K1_Robot_Project` | Gavinw575/K1_Robot_Project | master | `048bcf8` | clean |
| `booster/booster_assets` | BoosterRobotics/booster_assets | main | `486da0f` | untracked `K1_locomotion.xml` |
| `booster/booster_robotics_sdk` | BoosterRobotics/booster_robotics_sdk | main | `324946e` | patch (CMakeLists + binding.cpp) |
| `booster/booster_deploy` | BoosterRobotics/booster_deploy | main | `563e7c3` | patch + untracked `k1_velocity.py` |
| `booster/booster_train` | BoosterRobotics/booster_train | main | `b77e0e9` | patch + untracked velocity task |
| `experiments/navila` | **Janga786/k1-vlm-navigation** (private) | paper-audit-fixes | `fcf64e4` | full repo (cloned directly) |

> Third-party code is **not** re-hosted — only your diffs travel, as patches against the public
> upstreams. If a patch doesn't apply cleanly (upstream moved), use `git apply --3way`.

---

## 3. Trained locomotion policies (bundled — irreplaceable)

The RL locomotion policies are the one artifact that is **not** re-downloadable and **not** on
any upstream. They're small (5.7 MB each), so the K1 velocity / VLN-CE runs (748 MB) are bundled
here, split into <100 MB parts to satisfy GitHub's file-size limit.

```bash
bash migration/bootstrap.sh policies     # cat parts -> verify sha256 -> extract into booster/booster_train/logs/
```

Included: `k1_velocity`, `k1_velocity_vlnce`, `k1_velocity_vlnce_v3` (incl. the exported `.onnx`
the robot deploy loads). Excluded (unrelated to NaVILA, ~380 MB): `k1_fight_001`, `k1_mj_dance_004`
— if you want them, `rsync` `booster_train/logs/` from SOURCE when both machines share a network.

---

## 4. Build the `navila` env + download weights

```bash
bash migration/bootstrap.sh env-navila   # python 3.10.20, torch cu128, deps, editable vila/sdk, s2wrapper
bash migration/bootstrap.sh weights      # huggingface-cli download a8cheng/navila-llama3-8b-8f (~16GB)
```

Notes baked into the script:
- **torch is installed first** from `https://download.pytorch.org/whl/cu128` so nothing downgrades it.
- **No flash-attn / xformers** — your NaVILA patch makes them optional; inference uses `attn_implementation="sdpa"`. Do **not** install flash-attn for the inference path.
- **Editable installs use the correct TARGET path.** On SOURCE the `vila` editable wrongly pointed at
  `/home/janga/Projects/booster/NaVILA` (missing `k1_research`), forcing a `sys.path.insert` hack.
  Here it's installed from `~/Projects/k1_research/booster/NaVILA` — the hack is no longer needed.
- **ROS pip lines are filtered** from `navila.requirements.txt` (they're provided by `/opt/ros/jazzy`).
- Recipes for reference: `migration/env/navila.{environment.yml,requirements.txt,from-history.yml}`.

---

## 5. Full Isaac training / benchmark stack (full clone)

```bash
bash migration/bootstrap.sh env-isaac    # prints the guided sequence
```

Good news from SOURCE recon: **Isaac Sim is a pip package** (`isaacsim==5.1.0`), and IsaacLab is a
plain git checkout (`v2.3.0`). So the heavy stack is reproducible without a manual binary download:

```bash
conda create -y -n isaac_sim python=3.11 && conda activate isaac_sim
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && git checkout v2.3.0 && ./isaaclab.sh --install
# then recreate the project envs from the committed recipes:
conda env create -n vlnce-isaac -f ~/Projects/k1_research/migration/env/vlnce-isaac.environment.yml
# (same for isaacLab, isaacLab_311, booster, vla)
# NaVILA-Bench scene data (~5.2GB):
huggingface-cli download Zhaojing/VLN-CE-Isaac --repo-type dataset \
    --local-dir ~/Projects/k1_research/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/
```

The env recipe `.yml`/`.txt` files in `migration/env/` are exact pins from SOURCE — use them to
resolve any version drift. Benchmark `eval_results/` (52 GB) are **not** bundled — regenerate by
re-running NaVILA-Bench, or `rsync` from SOURCE if you want the originals.

---

## 6. Configure & verify

```bash
# Fix machine-specific values in the nav launch scripts (cloned into experiments/navila):
#   launch_demo.sh / stop_demo.sh:  DESKTOP_IP/HOST -> this TARGET's IP, ROBOT_IP -> the K1's LAN IP,
#   and use SSH keys instead of the embedded passwords.
export GROQ_API_KEY="<your-rotated-key>"

bash migration/bootstrap.sh verify
# then the definitive end-to-end check:
cd ~/Projects/k1_research/experiments/navila && python test_navila.py
# start the server:  ./start_navila_server.sh   (binds the NaVILA inference server)
```

`verify` confirms torch sees the 3090, `vila`/`llava`/`transformers` import, and the weights +
policies are present.

---

## What travels how (summary)

| Bucket | Items | Channel |
|---|---|---|
| **Git (private GitHub)** | all source, your local patches, env recipes, docs, **trained policies (split bundle)** | `k1-research-workspace` + `k1-vlm-navigation` |
| **Re-download (public)** | NaVILA weights (16 GB, HF), VLN-CE-Isaac scenes (5.2 GB, HF), Isaac Sim (pip), IsaacLab, all pip/apt deps, s2wrapper | network, on TARGET |
| **Regenerate / skip** | NaVILA-Bench `eval_results` (52 GB), OpenVLA cache (15 GB, unused) | not transferred |
| **Optional rsync** | full checkpoint history, unrelated fight/dance policies | only if SOURCE↔TARGET share a network |

Nothing required a USB drive or an 80 GB zip — code + small irreplaceables over GitHub, everything
else re-pulled from its canonical source.
