#!/bin/bash -l
#PBS -A UIC-HPC
#PBS -q by-gpu
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:eagle
#PBS -N sophia_import_test
#PBS -j oe

set -euo pipefail
cd "$PBS_O_WORKDIR"

# Write a live log file you can tail, regardless of PBS buffering
LOG="$PBS_O_WORKDIR/sophia_import_test.$PBS_JOBID.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== host: $(hostname) ==="
echo "=== jobid: $PBS_JOBID ==="
echo "=== workdir: $PBS_O_WORKDIR ==="
date

source envScriptSophia.sh

nvidia-smi || true
which python
python -c "import sys; print('python:', sys.executable)"

export TORCH_EXTENSIONS_DIR="/lus/eagle/projects/UIC-HPC/mrest/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "=== import test ==="
export PYTHONUNBUFFERED=1
python -u - <<'PY'
import torch
print("torch:", torch.__version__, torch.__file__, flush=True)
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count(), flush=True)

import framework_allreduce_zfp_cuda as ext
print("ext:", getattr(ext, "__file__", None), flush=True)
PY

echo "=== done ==="
date
