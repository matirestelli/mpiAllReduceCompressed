#ifndef cuZFP_h
#define cuZFP_h

#include "zfp.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
  size_t cuda_compress(zfp_stream *stream, const zfp_field *field);
  void cuda_decompress(zfp_stream *stream, zfp_field *field);

  //NEW ADDITION: adding this functions to the zfp library so it is possible to launch compress and cuda_decompress
  // kernels exacly on a specified non-default stream
  size_t cuda_compress_stream(
    zfp_stream *stream,
    const zfp_field *field,
    cudaStream_t cuda_stream);

  void cuda_decompress_stream(
      zfp_stream *stream,
      zfp_field *field,
      cudaStream_t cuda_stream);
      
#ifdef __cplusplus
}
#endif

#endif
