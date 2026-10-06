"""FAKE module path; executes the REAL NaVILA-Bench eval_utils.py (read-only) in this namespace."""
import os
_REAL = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py")
exec(compile(open(_REAL).read(), _REAL, "exec"))
