"""
Build script for the ring_allreduce_cuda extension.

Requirements:
    - CUDA toolkit installed
    - CUDA-aware MPI (OpenMPI or MPICH built with CUDA support)
    - PyTorch with CUDA support

Usage:
    cd hooks
    python setup.py install

Or for development (editable, no install):
    cd hooks
    python setup.py build_ext --inplace

Environment variables you may need to set:
    MPI_HOME   — path to MPI installation (e.g. /usr/local/mpi)
                 default: tries to find via mpi4py
    CUDA_HOME  — path to CUDA toolkit (e.g. /usr/local/cuda)
                 default: uses torch.utils.cpp_extension default
"""

import os
import subprocess
import sys

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def find_mpi_home():
    """
    Try to locate MPI installation directory.
    Priority: MPI_HOME env var > MPICH_DIR > mpi4py > common paths.
    """
    # 1. explicit MPI_HOME env var (set by user or module system)
    if "MPI_HOME" in os.environ:
        return os.environ["MPI_HOME"]

    # 2. MPICH_DIR (used by some Cray/HPC module systems)
    if "MPICH_DIR" in os.environ:
        return os.environ["MPICH_DIR"]

    # 3. ask mpi4py
    try:
        import mpi4py
        mpi_home = os.path.dirname(os.path.dirname(mpi4py.get_include()))
        if os.path.exists(os.path.join(mpi_home, "include", "mpi.h")):
            return mpi_home
    except ImportError:
        pass

    # 4. ask mpicc --showme:prefix (OpenMPI)
    try:
        result = subprocess.run(
            ["mpicc", "--showme:prefix"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 5. ask mpicc -show and parse -I flag (works for Cray MPICH)
    try:
        result = subprocess.run(
            ["mpicc", "-show"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for token in result.stdout.split():
                if token.startswith("-I"):
                    include_dir = token[2:]
                    candidate = os.path.dirname(include_dir)
                    if os.path.exists(os.path.join(include_dir, "mpi.h")):
                        return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 6. common system paths
    for path in ["/usr/local/mpi", "/usr/lib/openmpi", "/usr/lib/mpich",
                 "/opt/openmpi", "/opt/mpich"]:
        if os.path.exists(os.path.join(path, "include", "mpi.h")):
            return path

    raise RuntimeError(
        "Could not find MPI installation.\n"
        "Set MPI_HOME to your MPI prefix, e.g.:\n"
        "  MPI_HOME=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3 python setup.py install\n"
        "The prefix must contain include/mpi.h and lib/libmpi.so (or similar)."
    )


def find_mpi_include_and_lib(mpi_home: str):
    """
    Given an MPI home directory, find the actual include and lib paths.
    Handles standard layout (include/, lib/) and Cray MPICH which may
    have the headers directly in mpi_home or in subdirectories.
    """
    # candidate include dirs in priority order
    include_candidates = [
        os.path.join(mpi_home, "include"),
        mpi_home,                                          # Cray: mpi.h directly in MPI_HOME
        os.path.join(mpi_home, "include", "cray_mpich"),  # some Cray layouts
    ]
    # candidate lib dirs
    lib_candidates = [
        os.path.join(mpi_home, "lib"),
        os.path.join(mpi_home, "lib64"),
        mpi_home,
    ]

    mpi_include = None
    for c in include_candidates:
        if os.path.exists(os.path.join(c, "mpi.h")):
            mpi_include = c
            break

    mpi_lib = None
    for c in lib_candidates:
        # look for libmpi.so or libmpi_cray.so or libmpich.so
        for libname in ["libmpi.so", "libmpi_cray.so", "libmpich.so",
                        "libmpi.a", "libmpich.a"]:
            if os.path.exists(os.path.join(c, libname)):
                mpi_lib = c
                break
        if mpi_lib:
            break

    if mpi_include is None:
        raise RuntimeError(
            f"mpi.h not found under {mpi_home}.\n"
            f"Checked: {include_candidates}\n"
            f"Set MPI_HOME to the directory that contains include/mpi.h."
        )
    if mpi_lib is None:
        # fall back to lib/ even if we couldn't verify — linker will tell us
        mpi_lib = os.path.join(mpi_home, "lib")
        print(f"[setup] WARNING: could not find libmpi under {mpi_home}, "
              f"trying {mpi_lib} anyway")

    return mpi_include, mpi_lib


def main():
    mpi_home = find_mpi_home()
    mpi_include, mpi_lib = find_mpi_include_and_lib(mpi_home)

    print(f"[setup] Using MPI from : {mpi_home}")
    print(f"[setup] MPI include    : {mpi_include}")
    print(f"[setup] MPI lib        : {mpi_lib}")

    # Cray MPICH uses libmpi_cray or libmpich instead of libmpi
    # detect which one exists
    mpi_libname = "mpi"
    for candidate in ["mpi_cray", "mpich", "mpi"]:
        for ext in [".so", ".a"]:
            if os.path.exists(os.path.join(mpi_lib, f"lib{candidate}{ext}")):
                mpi_libname = candidate
                break

    print(f"[setup] MPI library    : lib{mpi_libname}")

    ext = CUDAExtension(
        name="ring_allreduce_cuda_ext",
        sources=["ring_allreduce_cuda.cu"],
        include_dirs=[mpi_include],
        library_dirs=[mpi_lib],
        libraries=[mpi_libname],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                f"-I{mpi_include}",
            ],
        },
        extra_link_args=[f"-Wl,-rpath,{mpi_lib}"],
    )

    setup(
        name="ring_allreduce_cuda_ext",
        version="0.1.0",
        description="Ring AllReduce via CUDA-aware MPI",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    main()