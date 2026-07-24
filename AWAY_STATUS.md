# AWAY_STATUS — lab machine handoff (compiled 2026-07-24; user away 2026-07-25 .. 08-23)

## Remote access
- SSH over Tailscale: `ssh boosterk1@100.116.68.68` (survives reboot; sshd + tailscaled enabled).
- LAN fallback: 10.218.0.139.

## GPU / driver situation (RESOLVED 2026-07-24)
- An auto-upgrade bumped the NVIDIA driver 580.159.03 -> 580.173.02 and broke CUDA (module
  mismatch). Recovered by rebuilding the 580.173.02 module from source with gcc-12 (DKMS).
  Machine now runs **driver 580.173.02**, CUDA verified working (torch.cuda.is_available()=True).
- **Recurrence is prevented for the trip:** auto-upgrade timers DISABLED
  (`apt-daily.timer`, `apt-daily-upgrade.timer`); 200 nvidia/cuda/kernel packages HELD.

## RELEASE THE HOLDS + RE-ENABLE UPDATES (run this WHEN YOU'RE BACK, not before):
```
sudo apt-mark unhold $(apt-mark showhold)
sudo systemctl enable --now apt-daily.timer apt-daily-upgrade.timer
```

## If CUDA breaks again while away (blind recovery over SSH):
```
# 1) confirm the symptom
python -c "import torch; print(torch.cuda.is_available())"          # False => broken
cat /proc/driver/nvidia/version                                     # loaded module version
# 2) rebuild the module for the running kernel with gcc-12 (scoped; no persistent cc symlink)
sudo apt-get install -y gcc-12
mkdir -p /tmp/g12 && ln -sf /usr/bin/gcc-12 /tmp/g12/gcc && ln -sf /usr/bin/gcc-12 /tmp/g12/cc
sudo env "PATH=/tmp/g12:/usr/sbin:/usr/bin:/sbin:/bin" dkms install nvidia/$(dkms status nvidia | grep -oE '580\.[0-9.]+' | head -1) -k $(uname -r) --force
rm -rf /tmp/g12
sudo reboot            # loads the rebuilt module; sshd+tailscale come back automatically
```
Notes: Secure Boot is OFF (unsigned DKMS module loads fine). nouveau is not blacklisted and
stays unloaded. The box boots with display + GPU both up.

## Code state (all committed + pushed)
- Eval fixes applied (navila_eval_v3.py): wall_timeout capture, term_reason/hit_step_cap,
  cam_z/cam_height/cam_aperture flags. Aggregator: -1.0 distance sentinel handling.
- Repos: github.com/Janga786/{k1-navila-research, NaVILA-Complete-Archive, booster_train,
  booster_deploy, k1-vlm-navigation}.

## EXPERIMENT STATUS: NOT YET LAUNCHED (pending acceptance/isolation tests + user go)
See receipts/PRE_REGISTRATION.md for the fixed analysis plan, the determinism verdict
(NON-deterministic), and the driver-change confound.
