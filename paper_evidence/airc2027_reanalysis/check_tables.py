"""Compare freshly computed tables with the shipped ones.

Usage: python check_tables.py SHIPPED_DIR NEW_DIR

Every JSON file in SHIPPED_DIR must exist in NEW_DIR with the same structure. Integers, strings,
booleans and nulls must be equal; floats must agree to a relative tolerance of 1e-9 (absolute 1e-12),
because the last floating-point digit can differ between CPUs and library versions.
Exit code 0 = all tables agree.
"""
import json
import math
import os
import sys

REL, ABS = 1e-9, 1e-12


def compare(a, b, path, problems):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b:
            problems.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                problems.append(f"{path}: {a} != {b}")
        elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            pass
        elif not math.isclose(a, b, rel_tol=REL, abs_tol=ABS):
            problems.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            problems.append(f"{path}: keys differ: {sorted(set(a) ^ set(b))}")
        for k in a:
            if k in b:
                compare(a[k], b[k], f"{path}.{k}", problems)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", problems)
    else:
        problems.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")


def main(shipped, new):
    bad = 0
    for name in sorted(f for f in os.listdir(shipped) if f.endswith(".json")):
        p = os.path.join(new, name)
        if not os.path.exists(p):
            print(f"MISSING  {name}")
            bad += 1
            continue
        with open(os.path.join(shipped, name)) as f1, open(p) as f2:
            a, b = json.load(f1), json.load(f2)
        problems = []
        compare(a, b, name, problems)
        with open(os.path.join(shipped, name), "rb") as f1, open(p, "rb") as f2:
            same_bytes = f1.read() == f2.read()
        status = "IDENTICAL" if same_bytes else ("AGREE" if not problems else "DIFFER")
        print(f"{status:9s} {name}")
        for line in problems[:20]:
            print("    " + line)
        bad += bool(problems)
    print("all tables agree" if bad == 0 else f"{bad} table(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
