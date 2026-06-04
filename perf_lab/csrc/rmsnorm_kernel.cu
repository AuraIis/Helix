#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ weight,
    scalar_t* __restrict__ y,
    float* __restrict__ inv_rms,
    int rows,
    int dim,
    float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum_sq = 0.0f;
  int base = row * dim;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float v = static_cast<float>(x[base + col]);
    sum_sq += v * v;
  }
  smem[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv = rsqrtf(smem[0] / static_cast<float>(dim) + eps);
  if (threadIdx.x == 0) {
    inv_rms[row] = inv;
  }

  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float out = static_cast<float>(x[base + col]) * inv * weight[col];
    y[base + col] = static_cast<scalar_t>(out);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_backward_x_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_x,
    int rows,
    int dim) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  int base = row * dim;
  float dot = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    dot += static_cast<float>(grad_out[base + col]) * weight[col] * static_cast<float>(x[base + col]);
  }
  smem[threadIdx.x] = dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv = inv_rms[row];
  float coeff = smem[0] * inv * inv * inv / static_cast<float>(dim);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    float go = static_cast<float>(grad_out[base + col]);
    float xv = static_cast<float>(x[base + col]);
    float gx = go * weight[col] * inv - xv * coeff;
    grad_x[base + col] = static_cast<scalar_t>(gx);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_backward_weight_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_weight,
    int rows,
    int dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= dim) {
    return;
  }
  float acc = 0.0f;
  for (int row = 0; row < rows; ++row) {
    int idx = row * dim + col;
    acc += static_cast<float>(grad_out[idx]) * static_cast<float>(x[idx]) * inv_rms[row];
  }
  grad_weight[col] = acc;
}

static int threads_for_dim(int dim) {
  if (dim >= 1024) return 256;
  if (dim >= 512) return 128;
  return 64;
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  auto y = torch::empty_like(x);
  int64_t dim64 = x.size(-1);
  int64_t rows64 = x.numel() / dim64;
  auto inv_rms = torch::empty({rows64}, x.options().dtype(torch::kFloat32));
  int rows = static_cast<int>(rows64);
  int dim = static_cast<int>(dim64);
  int threads = threads_for_dim(dim);
  size_t shared = threads * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rmsnorm_forward_cuda", [&] {
    rmsnorm_forward_kernel<scalar_t><<<rows, threads, shared, stream>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        y.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        rows,
        dim,
        static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, inv_rms};
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms) {
  auto grad_x = torch::empty_like(x);
  auto grad_weight = torch::empty_like(weight);
  int64_t dim64 = x.size(-1);
  int64_t rows64 = x.numel() / dim64;
  int rows = static_cast<int>(rows64);
  int dim = static_cast<int>(dim64);
  int threads = threads_for_dim(dim);
  size_t shared = threads * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rmsnorm_backward_cuda", [&] {
    rmsnorm_backward_x_kernel<scalar_t><<<rows, threads, shared, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<scalar_t>(),
        rows,
        dim);
    int w_threads = 256;
    int w_blocks = (dim + w_threads - 1) / w_threads;
    rmsnorm_backward_weight_kernel<scalar_t><<<w_blocks, w_threads, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        rows,
        dim);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_weight};
}

