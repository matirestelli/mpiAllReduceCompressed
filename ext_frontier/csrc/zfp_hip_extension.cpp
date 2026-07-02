#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>   // provides hipStream_t, hipMemcpyAsync, etc.


#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
#include <zfp.h>

// You must provide these in your HIP-modified zfp backend (analogous to cuda_*_stream).
size_t zfp_internal_hip_compress_stream(
    zfp_stream* stream,
    const zfp_field* field,
    hipStream_t hip_stream);

void zfp_internal_hip_decompress_stream(
    zfp_stream* stream,
    zfp_field* field,
    hipStream_t hip_stream);
}

namespace {

zfp_type zfpTypeFor(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32: return zfp_type_float;
    case torch::kFloat64: return zfp_type_double;
    default:
      throw std::runtime_error("ZFP HIP hook supports only float32/float64 tensors");
  }
}

void validateInputTensor(const torch::Tensor& tensor, const char* name) {
  if (!tensor.is_cuda()) { // note: ROCm tensors still report is_cuda() == True in PyTorch
    throw std::runtime_error(std::string(name) + " must be a CUDA/HIP tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
  if (tensor.dim() != 1) {
    throw std::runtime_error(std::string(name) + " must be a flattened 1D tensor");
  }
}

void setHipFixedRate(zfp_stream* zfp, const torch::Tensor& tensor, double rate) {
  const zfp_type type = zfpTypeFor(tensor);
  zfp_stream_set_rate(zfp, rate, type, 1, 0);

  if (!zfp_stream_set_execution(zfp, zfp_exec_hip)) {
    throw std::runtime_error(
        "Failed to set zfp execution policy to HIP. "
        "Ensure libzfp was built with HIP support enabled.");
  }
}

hipStream_t currentHipStreamFor(const torch::Tensor& tensor) {
  const int device_index = tensor.get_device();
  // Under ROCm PyTorch, this returns the current HIP stream.
  return (hipStream_t)at::hip::getCurrentHIPStream(device_index).stream();
}

} // namespace

int64_t max_output_bytes(torch::Tensor input, double rate) {
  validateInputTensor(input, "input");

  zfp_field* field = zfp_field_1d(input.data_ptr(), zfpTypeFor(input), input.numel());
  if (!field) throw std::runtime_error("zfp_field_1d failed");

  bitstream* bs = stream_open(nullptr, 0);
  if (!bs) { zfp_field_free(field); throw std::runtime_error("stream_open failed"); }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); zfp_field_free(field); throw std::runtime_error("zfp_stream_open failed"); }

  setHipFixedRate(zfp, input, rate);
  size_t max_bytes = zfp_stream_maximum_size(zfp, field);

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
  return static_cast<int64_t>(max_bytes);
}

int64_t compress_into(torch::Tensor input, torch::Tensor output, double rate) {
  validateInputTensor(input, "input");
  validateInputTensor(output, "output");
  if (output.scalar_type() != torch::kUInt8)
    throw std::runtime_error("output must be a uint8 CUDA/HIP tensor");

  zfp_field* field = zfp_field_1d(input.data_ptr(), zfpTypeFor(input), input.numel());
  if (!field) throw std::runtime_error("zfp_field_1d failed");

  bitstream* bs = stream_open(output.data_ptr(), output.numel());
  if (!bs) { zfp_field_free(field); throw std::runtime_error("stream_open failed"); }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); zfp_field_free(field); throw std::runtime_error("zfp_stream_open failed"); }

  setHipFixedRate(zfp, input, rate);
  zfp_stream_rewind(zfp);

  size_t used_bytes = zfp_compress(zfp, field);
  if (used_bytes == 0) {
    zfp_stream_close(zfp);
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("zfp_compress failed");
  }

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
  return static_cast<int64_t>(used_bytes);
}

void decompress_into(torch::Tensor input, int64_t used_bytes, torch::Tensor output, double rate) {
  validateInputTensor(input, "input");
  validateInputTensor(output, "output");
  if (input.scalar_type() != torch::kUInt8)
    throw std::runtime_error("input must be a uint8 CUDA/HIP tensor");
  if (used_bytes < 0 || used_bytes > input.numel())
    throw std::runtime_error("used_bytes exceeds compressed buffer capacity");

  zfp_field* field = zfp_field_1d(output.data_ptr(), zfpTypeFor(output), output.numel());
  if (!field) throw std::runtime_error("zfp_field_1d failed");

  bitstream* bs = stream_open(input.data_ptr(), static_cast<size_t>(used_bytes));
  if (!bs) { zfp_field_free(field); throw std::runtime_error("stream_open failed"); }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); zfp_field_free(field); throw std::runtime_error("zfp_stream_open failed"); }

  setHipFixedRate(zfp, output, rate);
  zfp_stream_rewind(zfp);

  if (!zfp_decompress(zfp, field)) {
    zfp_stream_close(zfp);
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("zfp_decompress failed");
  }

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
}

// stream-aware
int64_t compress_into_current_stream(torch::Tensor input, torch::Tensor output, double rate) {
  validateInputTensor(input, "input");
  validateInputTensor(output, "output");
  if (output.scalar_type() != torch::kUInt8)
    throw std::runtime_error("output must be a uint8 CUDA/HIP tensor");

  zfp_field* field = zfp_field_1d(input.data_ptr(), zfpTypeFor(input), input.numel());
  if (!field) throw std::runtime_error("zfp_field_1d failed");

  bitstream* bs = stream_open(output.data_ptr(), output.numel());
  if (!bs) { zfp_field_free(field); throw std::runtime_error("stream_open failed"); }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); zfp_field_free(field); throw std::runtime_error("zfp_stream_open failed"); }

  setHipFixedRate(zfp, input, rate);
  zfp_stream_rewind(zfp);

  hipStream_t hip_stream = currentHipStreamFor(input);
  size_t used_bytes = zfp_internal_hip_compress_stream(zfp, field, hip_stream);

  if (used_bytes == 0) {
    zfp_stream_close(zfp);
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("hip_compress_stream failed");
  }

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
  return static_cast<int64_t>(used_bytes);
}

void decompress_into_current_stream(torch::Tensor input, int64_t used_bytes, torch::Tensor output, double rate) {
  validateInputTensor(input, "input");
  validateInputTensor(output, "output");
  if (input.scalar_type() != torch::kUInt8)
    throw std::runtime_error("input must be a uint8 CUDA/HIP tensor");
  if (used_bytes < 0 || used_bytes > input.numel())
    throw std::runtime_error("used_bytes exceeds compressed buffer capacity");

  zfp_field* field = zfp_field_1d(output.data_ptr(), zfpTypeFor(output), output.numel());
  if (!field) throw std::runtime_error("zfp_field_1d failed");

  bitstream* bs = stream_open(input.data_ptr(), static_cast<size_t>(used_bytes));
  if (!bs) { zfp_field_free(field); throw std::runtime_error("stream_open failed"); }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); zfp_field_free(field); throw std::runtime_error("zfp_stream_open failed"); }

  setHipFixedRate(zfp, output, rate);
  zfp_stream_rewind(zfp);

  hipStream_t hip_stream = currentHipStreamFor(output);
  zfp_internal_hip_decompress_stream(zfp, field, hip_stream);

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_output_bytes", &max_output_bytes, "ZFP HIP max compressed bytes");

  m.def("compress_into", &compress_into,
        "ZFP HIP compress into preallocated buffer using original path");
  m.def("decompress_into", &decompress_into,
        "ZFP HIP decompress into preallocated buffer using original path");

  m.def("compress_into_current_stream", &compress_into_current_stream,
        "ZFP HIP compress into preallocated buffer on current PyTorch stream");
  m.def("decompress_into_current_stream", &decompress_into_current_stream,
        "ZFP HIP decompress into preallocated buffer on current PyTorch stream");
}
