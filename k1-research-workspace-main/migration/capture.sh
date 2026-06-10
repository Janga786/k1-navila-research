#!/usr/bin/env bash
# capture.sh — snapshot everything needed to reproduce the NaVILA / K1 workspace
# on a new machine. Produces patches, a repo manifest, untracked-file copies,
# env recipes, and a trained-policy bundle under ./migration/.
#
# SAFE: read-only against the source repos; only writes under migration/.
set -uo pipefail

ROOT="/home/janga/Projects/k1_research"
MIG="$ROOT/migration"
mkdir -p "$MIG"/{patches,untracked,env,checkpoints}

manifest="$MIG/repos.manifest.tsv"
printf 'name\tpath\tupstream\tbranch\tcommit\tdirty_files\n' > "$manifest"

# repo<TAB>relpath  (upstream/branch/commit are read live from each repo)
repos=(
  "navila|booster/NaVILA"
  "navila-bench|NaVILA-Bench"
  "legged-loco|legged-loco"
  "k1-robot-project|K1_Robot_Project"
  "booster_assets|booster/booster_assets"
  "booster_robotics_sdk|booster/booster_robotics_sdk"
  "booster_deploy|booster/booster_deploy"
  "booster_train|booster/booster_train"
  "k1-vlm-navigation|experiments/navila"
  "workspace|."
)

# directories whose untracked content is huge / regenerable — never capture
EXCLUDE_RE='(^|/)(eval_results|out|out_[^/]*|logs|__pycache__|\.egg-info|\.pytest_cache|wandb|videos|\.mp4$|\.bin$|\.safetensors$)'

for entry in "${repos[@]}"; do
  name="${entry%%|*}"; rel="${entry##*|}"; dir="$ROOT/$rel"
  [ -d "$dir/.git" ] || { echo "SKIP $name ($rel: no .git)"; continue; }
  up=$(git -C "$dir" remote get-url origin 2>/dev/null || echo "(none)")
  br=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null)
  sha=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  ndirty=$(git -C "$dir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rel" "$up" "$br" "$sha" "$ndirty" >> "$manifest"

  # 1) tracked-file modifications -> patch
  if ! git -C "$dir" diff --quiet 2>/dev/null; then
    git -C "$dir" diff > "$MIG/patches/$name.patch"
    echo "patch: $name ($(wc -l < "$MIG/patches/$name.patch") lines)"
  fi

  # 2) untracked files worth keeping (small, non-regenerable) -> tar, preserving paths
  mapfile -t untracked < <(git -C "$dir" ls-files --others --exclude-standard 2>/dev/null \
      | grep -vE "$EXCLUDE_RE")
  if [ "${#untracked[@]}" -gt 0 ]; then
    # drop any single file >50MB defensively
    keep=()
    for f in "${untracked[@]}"; do
      sz=$(stat -c%s "$dir/$f" 2>/dev/null || echo 0)
      [ "$sz" -le 52428800 ] && keep+=("$f") || echo "  skip big untracked: $rel/$f ($((sz/1048576))MB)"
    done
    if [ "${#keep[@]}" -gt 0 ]; then
      tar -C "$dir" -czf "$MIG/untracked/$name.untracked.tgz" "${keep[@]}" 2>/dev/null \
        && echo "untracked: $name (${#keep[@]} files)"
    fi
  fi
done

echo
echo "=== repos.manifest.tsv ==="
column -t -s$'\t' "$manifest"
echo
echo "=== patches ==="; ls -la "$MIG/patches" 2>/dev/null
echo "=== untracked bundles ==="; ls -la "$MIG/untracked" 2>/dev/null