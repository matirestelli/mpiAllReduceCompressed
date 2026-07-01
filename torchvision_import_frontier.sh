#!/bin/bash
# Install torchvision FROM SOURCE against an already-installed ROCm PyTorch.
# Usage:
#   source ./env_frontier_login.sh
#   bash ./install_torchvision_rocm.sh
#
# Optional:
#   export TORCHVISION_SRC=/path/to/vision            # if set, use existing local clone
#   export TORCHVISION_REF=v0.24.0                   # tag/branch/commit to clone when TORCHVISION_SRC not set
#   export TORCHVISION_WORK=/path/on/lustre/vision    # where to clone+patch (recommended on Lustre)
set -euo pipefail

LOG="/tmp/${USER}_torchvision_install_rocm_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"

(
  set -x

  # Force a modern compiler
  export CC=/opt/cray/pe/gcc-native/14/bin/gcc
  export CXX=/opt/cray/pe/gcc-native/14/bin/g++
  export CMAKE_C_COMPILER="$CC"
  export CMAKE_CXX_COMPILER="$CXX"

  # These may or may not be honored by the ref; we keep them anyway.
  export BUILD_TORCHVISION_OPS=0
  export TORCHVISION_DISABLE_EXTENSIONS=1
  export FORCE_CUDA=0
  export USE_CUDA=0
  export USE_ROCM=1

  command -v python
  python -V
  python -m pip -V

  command -v "$CC"
  command -v "$CXX"
  "$CXX" --version

  python - <<'PY'
import sys, torch
print("torch:", torch.__version__, "from", torch.__file__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("cuda.is_available:", torch.cuda.is_available())  # may be False on login nodes
if getattr(torch.version, "hip", None) in (None, ""):
    print("ERROR: torch is not a ROCm build (torch.version.hip is empty). Refusing.", file=sys.stderr)
    sys.exit(2)
PY

  python -m pip uninstall -y torchvision >/dev/null 2>&1 || true
  python -m pip install -U wheel "setuptools<82"

  # Prepare source tree (local or clone)
  if [[ -n "${TORCHVISION_SRC:-}" ]]; then
    TV_DIR="$TORCHVISION_SRC"
  else
    TV_DIR="${TORCHVISION_WORK:-$PWD/torchvision_src_${TORCHVISION_REF}}"
    rm -rf "$TV_DIR"
    git clone --depth 1 --branch "${TORCHVISION_REF}" https://github.com/pytorch/vision.git "$TV_DIR"
  fi

  cd "$TV_DIR"

  # ---- PATCH: avoid duplicate symbol vision::cuda_version() from vision.cpp and vision_hip.cpp ----
  # We give the HIP file's function internal linkage by inserting an anonymous namespace inside `namespace vision`.
  python - <<'PY'
import pathlib, re, sys

p = pathlib.Path("torchvision/csrc/vision_hip.cpp")
if not p.exists():
    print(f"WARNING: {p} not found; skipping patch.")
    sys.exit(0)

s = p.read_text()

if "cuda_version" not in s:
    print("NOTE: cuda_version not present in vision_hip.cpp; no patch needed.")
    sys.exit(0)

# If already patched, do nothing.
if "anonymous namespace" in s:
    print("NOTE: vision_hip.cpp already appears patched; skipping.")
    sys.exit(0)

# Insert anonymous namespace after first 'namespace vision {'
s2, n = re.subn(r"(namespace\s+vision\s*\{\s*)", r"\1\nnamespace {\n", s, count=1)
if n != 1:
    print("WARNING: Could not locate 'namespace vision {' uniquely; skipping patch.")
    sys.exit(0)

# Close anonymous namespace right before the final '}' in the file (best-effort).
last = s2.rfind("}")
if last == -1:
    print("WARNING: Could not find closing brace; skipping patch.")
    sys.exit(0)

s2 = s2[:last] + "\n} // anonymous namespace\n\n" + s2[last:]
p.write_text(s2)
print("Patched:", p)
PY
  # ---- END PATCH ----

  # Install from the patched local tree
  python -m pip install -v --no-build-isolation --no-deps -U .

  python -u - <<'PY'
import torch, torchvision
print("torch:", torch.__version__, "hip:", torch.version.hip)
print("torchvision:", torchvision.__version__, "from", torchvision.__file__)
PY

) >"$LOG" 2>&1

echo "---- last 200 lines ----"
tail -200 "$LOG"
echo "------------------------"

grep -q "torchvision:" "$LOG"
echo "OK: torchvision installed successfully."
