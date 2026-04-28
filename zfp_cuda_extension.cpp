#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <zfp.h>
}

namespace {

zfp_type zfpTypeFor(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return zfp_type_float;
    case torch::kFloat64:
      return zfp_type_double;
    default:
      throw std::runtime_error("ZFP CUDA hook supports only float32/float64 tensors");
  }
}

void validateInputTensor(const torch::Tensor& tensor, const char* name) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
  if (tensor.dim() != 1) {
    throw std::runtime_error(std::string(name) + " must be a flattened 1D tensor");
  }
}

void setCudaFixedRate(zfp_stream* zfp, const torch::Tensor& tensor, double rate) {
  const zfp_type type = zfpTypeFor(tensor);
  // The official CUDA backend supports fixed-rate mode only.
  zfp_stream_set_rate(zfp, rate, type, 1, 0);
  if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
    throw std::runtime_error(
        "Failed to set zfp execution policy to CUDA. "
        "Ensure libzfp was built with ZFP_WITH_CUDA=ON.");
  }
}

}  // namespace

int64_t max_output_bytes(torch::Tensor input, double rate) {
  validateInputTensor(input, "input");

  zfp_field* field = zfp_field_1d(input.data_ptr(), zfpTypeFor(input), input.numel());
  if (!field) {
    throw std::runtime_error("zfp_field_1d failed");
  }

  bitstream* bs = stream_open(nullptr, 0);
  if (!bs) {
    zfp_field_free(field);
    throw std::runtime_error("stream_open failed");
  }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("zfp_stream_open failed");
  }

  setCudaFixedRate(zfp, input, rate);
  size_t max_bytes = zfp_stream_maximum_size(zfp, field);

  zfp_stream_close(zfp);
  stream_close(bs);
  zfp_field_free(field);
  return static_cast<int64_t>(max_bytes);
}

int64_t compress_into(torch::Tensor input, torch::Tensor output, double rate) {
  validateInputTensor(input, "input");
  validateInputTensor(output, "output");
  if (output.scalar_type() != torch::kUInt8) {
    throw std::runtime_error("output must be a uint8 CUDA tensor");
  }

  zfp_field* field = zfp_field_1d(input.data_ptr(), zfpTypeFor(input), input.numel());
  if (!field) {
    throw std::runtime_error("zfp_field_1d failed");
  }

  bitstream* bs = stream_open(output.data_ptr(), output.numel());
  if (!bs) {
    zfp_field_free(field);
    throw std::runtime_error("stream_open failed");
  }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("zfp_stream_open failed");
  }

  setCudaFixedRate(zfp, input, rate);
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
  if (input.scalar_type() != torch::kUInt8) {
    throw std::runtime_error("input must be a uint8 CUDA tensor");
  }
  if (used_bytes < 0 || used_bytes > input.numel()) {
    throw std::runtime_error("used_bytes exceeds compressed buffer capacity");
  }

  zfp_field* field = zfp_field_1d(output.data_ptr(), zfpTypeFor(output), output.numel());
  if (!field) {
    throw std::runtime_error("zfp_field_1d failed");
  }

  bitstream* bs = stream_open(input.data_ptr(), static_cast<size_t>(used_bytes));
  if (!bs) {
    zfp_field_free(field);
    throw std::runtime_error("stream_open failed");
  }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    zfp_field_free(field);
    throw std::runtime_error("zfp_stream_open failed");
  }

  setCudaFixedRate(zfp, output, rate);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_output_bytes", &max_output_bytes, "ZFP CUDA max compressed bytes");
  m.def("compress_into", &compress_into, "ZFP CUDA compress into preallocated buffer");
  m.def("decompress_into", &decompress_into, "ZFP CUDA decompress into preallocated buffer");
}
