#!/bin/bash
# Scan files about to be committed for secrets (tokens, keys, passwords, .env-type files).
# Usage: secret_scan.sh <file>...   Prints every hit; exit 1 if any hit, 0 if clean.
# Text files: token formats + generic "password/secret/token/api_key = '...'" assignments.
# Binary files (npz, png, ...): strict token formats only.
hits=0
STRICT='(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[abprs]-[A-Za-z0-9-]{10,}|sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35})'
GENERIC='(password|passwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|bearer)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{6,}'
for f in "$@"; do
  b=$(basename "$f")
  case "$b" in
    .env|.env.*|*.pem|*.key|id_rsa*|id_ed25519*|.git-credentials|.netrc|*.p12|*.pfx)
      echo "SECRET-FILE: $f"; hits=$((hits+1)); continue;;
  esac
  if grep -Iq . "$f" 2>/dev/null; then   # text file
    if grep -nE "$STRICT" "$f"; then echo "  ^ token pattern in $f"; hits=$((hits+1)); fi
    if grep -niE "$GENERIC" "$f"; then echo "  ^ credential assignment in $f"; hits=$((hits+1)); fi
  else
    if LC_ALL=C grep -aqE "$STRICT" "$f"; then echo "TOKEN-PATTERN (binary): $f"; hits=$((hits+1)); fi
  fi
done
echo "[secret_scan] files=$# hits=$hits"
[ $hits -eq 0 ]
