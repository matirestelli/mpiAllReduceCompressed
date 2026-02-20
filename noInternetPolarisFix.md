Polaris compute nodes (like x3005c0s13b1n0) block direct internet for security/scaling, routing all outbound traffic through ALCF's proxy server at proxy.alcf.anl.gov:3128.

Why These Exports?
http_proxy / https_proxy: Tells pip/curl to use proxy for HTTP/HTTPS pypi.org downloads (your "Network unreachable" error).

ftp_proxy: Covers FTP mirrors (rare, but complete).

no_proxy: Excludes local/internal traffic (e.g., localhost, Polaris internal nets) from proxying—avoids loops/delays. polaris-* matches node hostnames.

Permanent Fix
Add to your polaris_baseline_env.sh:

text
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*,grand.alschub.org"

Then retry:

text
pip install --no-cache-dir ninja pyyaml setuptools cmake cffi typing_extensions sympy mkl mkl-include
Add these proxies to your polaris_baseline_env.sh script for persistence.
​

Script for mpi true:

 #!/bin/bash
module purge
module load PrgEnv-nvidia/8.6.0 cuda/12.9 craype-accel-nvidia80
module use /soft/modulefiles
module load conda/2025-09-28
conda activate base
export OMP_NUM_THREADS=1
mpiexec -n 4 -ppn 1 python baselineTraining.py
