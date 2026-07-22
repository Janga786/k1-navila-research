#!/usr/bin/env bash
# Per-repo authorship ledger. Vendored trees (NaVILA-Bench upstream, IsaacLab clone) are
# excluded from "authored LOC" and counted separately. Regenerate: bash receipts/git_ledger.sh
for R in ~/Projects/k1_research ~/robots/k1/workspace/booster_train ~/robots/k1/workspace/booster_deploy ~/robots/k1/workspace/k1-vlm-navigation; do
  echo "== $R =="
  git -C "$R" log --pretty='%h %ad %an %s' --date=short | tail -1 | awk '{print "first:", $2}'
  git -C "$R" log --pretty='%ad' --date=short | head -1 | awk '{print "last:", $1}'
  echo "commits: $(git -C "$R" rev-list --count HEAD)"
  git -C "$R" log --numstat --pretty= | awk '{a+=$1; d+=$2} END{print "LOC added:", a, "deleted:", d}'
  echo "uncommitted (working tree): $(git -C "$R" status --porcelain | wc -l) paths"
done
