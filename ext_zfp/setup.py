import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

ZFP_PREFIX = os.environ.get("ZFP_PREFIX", os.path.join(HERE, "..", "zfp-install"))
ZFP_INCLUDE = os.path.join(ZFP_PREFIX, "include")
ZFP_LIBDIR = os.path.join(ZFP_PREFIX, "lib64")  # Polaris uses lib64

ext = CUDAExtension(
    name="framework_allreduce_zfp_cuda",
    sources=[os.path.join(HERE, "csrc", "zfp_cuda_extension.cpp")],
    include_dirs=[ZFP_INCLUDE],
    library_dirs=[ZFP_LIBDIR],
    libraries=["zfp"],
    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    extra_link_args=[f"-Wl,-rpath,{ZFP_LIBDIR}"],
)

setup(
    name="framework_allreduce_zfp_cuda",
    version="0.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
