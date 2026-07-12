"""
model/cuda/kernels.py

Raw CUDA C source for the sm_35 (Kepler GT 730) PyCUDA SourceModule.
Precision is strictly float32.

All row-wise reductions (layernorm mean/var, softmax max/sum) use a
shared-memory tree reduction with an explicit zero/sentinel initialization
step before any accumulation, and require a POWER-OF-TWO block size
(enforced by model/cuda/ops.py). This avoids two common pitfalls:
  1. Reading uninitialized __shared__ accumulators via atomicAdd.
  2. Using atomicMax on a float-as-int cast, which is only monotonic for
     non-negative values and is unsafe for logits (which can be negative).
"""

CUDA_SOURCE = r"""
// sm_35 compatible Matrix Multiplication (C = A * B)
// A: [M, K] row-major, B: [K, N] row-major, C: [M, N] row-major
__global__ void gemm_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Elementwise add: out = a + b (broadcasts b over rows if b_len < n_elements and n_elements % b_len == 0)
__global__ void add_bias_fp32(const float* a, const float* bias, float* out, int n_elements, int bias_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        out[idx] = a[idx] + bias[idx % bias_len];
    }
}

// Layernorm: one block per row. blockDim.x MUST be a power of two.
__global__ void layernorm_fp32(const float* x, float* out, const float* gamma, const float* beta,
                                int hidden_dim, float eps, int total_rows) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int offset = row * hidden_dim;

    // 1. Mean via tree reduction
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        local_sum += x[offset + i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / hidden_dim;
    __syncthreads();

    // 2. Variance via tree reduction
    float local_var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = x[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    sdata[tid] = local_var_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / hidden_dim;
    __syncthreads();

    // 3. Normalize + scale/shift
    float inv_std = rsqrtf(variance + eps);
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = (x[offset + i] - mean) * inv_std;
        out[offset + i] = normalized * gamma[i] + beta[i];
    }
}

// GeLU (tanh approximation), elementwise
__global__ void gelu_fp32(const float* x, float* out, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
        out[idx] = val * cdf;
    }
}

// Softmax along the last dimension. One block per row. blockDim.x MUST be a power of two.
__global__ void softmax_fp32(const float* logits, float* probs, int vocab_size, int total_rows) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int offset = row * vocab_size;

    // 1. Row max via tree reduction (safe for negative logits)
    float local_max = -3.402823466e+38f; // -FLT_MAX
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // 2. Sum of exp(x - max) via tree reduction
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_sum += expf(logits[offset + i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // 3. Normalize
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        probs[offset + i] = expf(logits[offset + i] - row_max) / row_sum;
    }
}
"""
