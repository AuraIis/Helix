#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms);

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(x.size(-1) == weight.numel(), "last x dim must match weight");
  return rmsnorm_forward_cuda(x, weight, eps);
}

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms) {
  TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && weight.is_cuda() && inv_rms.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(grad_out.is_contiguous() && x.is_contiguous() && weight.is_contiguous(), "tensors must be contiguous");
  return rmsnorm_backward_cuda(grad_out, x, weight, inv_rms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Auralis RMSNorm forward (CUDA)");
  m.def("rmsnorm_backward", &rmsnorm_backward, "Auralis RMSNorm backward (CUDA)");
}

