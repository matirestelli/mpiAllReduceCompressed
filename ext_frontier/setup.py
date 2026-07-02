import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))

ZFP_PREFIX = os.environ.get("ZFP_PREFIX", os.path.join(HERE, "..", "zfp-install-frontier"))
ZFP_INCLUDE = os.path.join(ZFP_PREFIX, "include")
ZFP_LIBDIR64 = os.path.join(ZFP_PREFIX, "lib64")
ZFP_LIBDIR = os.path.join(ZFP_PREFIX, "lib")

# pick whichever exists (Frontier installs sometimes use lib, sometimes lib64)
ZFP_LIBDIR_USE = ZFP_LIBDIR64 if os.path.isdir(ZFP_LIBDIR64) else ZFP_LIBDIR

ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
ROCM_INCLUDE = os.path.join(ROCM_HOME, "include")
ROCM_LIBDIR64 = os.path.join(ROCM_HOME, "lib64")
ROCM_LIBDIR = os.path.join(ROCM_HOME, "lib")
ROCM_LIBDIR_USE = ROCM_LIBDIR64 if os.path.isdir(ROCM_LIBDIR64) else ROCM_LIBDIR

ext = CppExtension(
    name="framework_allreduce_zfp_hip",
    sources=[os.path.join(HERE, "csrc", "zfp_hip_extension.cpp")],  # <-- rename to your actual filename
    include_dirs=[ZFP_INCLUDE, ROCM_INCLUDE],
    library_dirs=[ZFP_LIBDIR_USE, ROCM_LIBDIR_USE],
    libraries=["zfp"],
    extra_compile_args=["-O3", "-std=c++17"],
    extra_link_args=[f"-Wl,-rpath,{ZFP_LIBDIR_USE}", f"-Wl,-rpath,{ROCM_LIBDIR_USE}"],
)

setup(
    name="framework_allreduce_zfp_hip",
    version="0.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
