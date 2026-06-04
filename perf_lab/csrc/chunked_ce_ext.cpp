#include <torch/extension.h>

#include <limits>
#include <vector>

namespace {

void check_inputs(const torch::Tensor& hidden,
                  const torch::Tensor& weight,
                  const torch::Tensor& labels,
                  int64_t chunk_size) {
  TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(labels.is_cuda(), "labels must be CUDA");
  TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [N, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [V, D]");
  TORCH_CHECK(labels.dim() == 1, "labels must be [N]");
  TORCH_CHECK(hidden.size(0) == labels.size(0), "labels length must match hidden rows");
  TORCH_CHECK(hidden.size(1) == weight.size(1), "hidden dim must match weight dim");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64, "labels must be int64");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");
}

}  // namespace

std::vector<torch::Tensor> chunked_ce_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t chunk_size,
    int64_t ignore_index) {
  torch::NoGradGuard no_grad;
  check_inputs(hidden, weight, labels, chunk_size);

  const int64_t tokens = hidden.size(0);
  const int64_t vocab_size = weight.size(0);
  auto float_opts = hidden.options().dtype(torch::kFloat32);

  auto valid = labels.ne(ignore_index);
  auto valid_count = valid.sum().clamp_min(1);
  auto row_max = torch::full(
      {tokens},
      -std::numeric_limits<float>::infinity(),
      float_opts);
  auto target_logits = torch::zeros({tokens}, float_opts);

  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t end = std::min(start + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start, end);
    auto logits = torch::matmul(hidden, weight_chunk.t()).to(torch::kFloat32);
    row_max = torch::maximum(row_max, std::get<0>(logits.max(1)));

    auto in_chunk = valid.logical_and(labels.ge(start).logical_and(labels.lt(end)));
    auto rows = torch::nonzero(in_chunk).flatten();
    auto cols = labels.index_select(0, rows) - start;
    auto vals = logits.index({rows, cols});
    target_logits.index_put_({rows}, vals);
  }

  auto exp_sum = torch::zeros({tokens}, float_opts);
  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t end = std::min(start + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start, end);
    auto logits = torch::matmul(hidden, weight_chunk.t()).to(torch::kFloat32);
    exp_sum += torch::exp(logits - row_max.unsqueeze(1)).sum(1);
  }

  auto losses = row_max + torch::log(exp_sum) - target_logits;
  losses = torch::where(valid, losses, torch::zeros_like(losses));
  auto loss = losses.sum() / valid_count.to(torch::kFloat32);
  return {loss, row_max, exp_sum, valid, valid_count};
}

std::vector<torch::Tensor> chunked_ce_backward(
    torch::Tensor grad_output,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    torch::Tensor row_max,
    torch::Tensor exp_sum,
    torch::Tensor valid,
    torch::Tensor valid_count,
    int64_t chunk_size,
    int64_t ignore_index) {
  torch::NoGradGuard no_grad;
  check_inputs(hidden, weight, labels, chunk_size);
  TORCH_CHECK(row_max.is_cuda() && exp_sum.is_cuda() && valid.is_cuda() && valid_count.is_cuda(),
              "saved tensors must be CUDA");

  const int64_t vocab_size = weight.size(0);
  auto hidden_float = hidden.to(torch::kFloat32);
  auto grad_hidden = torch::zeros_like(hidden_float);
  auto grad_weight = torch::zeros_like(weight);
  auto scale = grad_output.to(torch::kFloat32) / valid_count.to(torch::kFloat32);

  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t end = std::min(start + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start, end);
    auto logits = torch::matmul(hidden, weight_chunk.t()).to(torch::kFloat32);
    auto probs = torch::exp(logits - row_max.unsqueeze(1)) / exp_sum.unsqueeze(1);
    probs = torch::where(valid.unsqueeze(1), probs, torch::zeros_like(probs));

    auto in_chunk = valid.logical_and(labels.ge(start).logical_and(labels.lt(end)));
    auto rows = torch::nonzero(in_chunk).flatten();
    auto cols = labels.index_select(0, rows) - start;
    auto vals = probs.index({rows, cols}) - 1.0;
    probs.index_put_({rows, cols}, vals);

    auto grad_logits = probs * scale;
    grad_hidden += torch::matmul(grad_logits, weight_chunk.to(torch::kFloat32));
    auto grad_weight_chunk = torch::matmul(grad_logits.t(), hidden_float).to(weight.scalar_type());
    grad_weight.slice(0, start, end).copy_(grad_weight_chunk);
  }

  return {grad_hidden.to(hidden.scalar_type()), grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chunked_ce_forward", &chunked_ce_forward, "Auralis chunked linear CE forward (CUDA tensors, Torch ops)");
  m.def("chunked_ce_backward", &chunked_ce_backward, "Auralis chunked linear CE backward (CUDA tensors, Torch ops)");
}
