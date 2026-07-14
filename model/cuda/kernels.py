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
// Tiled GEMM for Kepler (sm_35). TILE=16 fits well on GT 730.
#define GEMM_TILE 16

__global__ void gemm_fp32(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[GEMM_TILE][GEMM_TILE];
    __shared__ float sB[GEMM_TILE][GEMM_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE + ty;
    int col = blockIdx.x * GEMM_TILE + tx;

    float sum = 0.0f;
    int tiles = (K + GEMM_TILE - 1) / GEMM_TILE;

    for (int t = 0; t < tiles; ++t) {
        int a_col = t * GEMM_TILE + tx;
        int b_row = t * GEMM_TILE + ty;

        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < GEMM_TILE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled GEMM + bias: C = A @ B + bias[col]
__global__ void gemm_bias_fp32(const float* __restrict__ A, const float* __restrict__ B,
                               const float* __restrict__ bias, float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float sA[GEMM_TILE][GEMM_TILE];
    __shared__ float sB[GEMM_TILE][GEMM_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE + ty;
    int col = blockIdx.x * GEMM_TILE + tx;

    float sum = 0.0f;
    int tiles = (K + GEMM_TILE - 1) / GEMM_TILE;

    for (int t = 0; t < tiles; ++t) {
        int a_col = t * GEMM_TILE + tx;
        int b_row = t * GEMM_TILE + ty;

        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < GEMM_TILE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
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

// Layernorm forward that also caches xhat and per-row invstd for backward on device.
__global__ void layernorm_cache_fp32(
    const float* x, float* y, float* xhat, float* invstd_row,
    const float* gamma, const float* beta,
    int hidden_dim, float eps, int total_rows
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int offset = row * hidden_dim;

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
    float inv_std = rsqrtf(variance + eps);
    if (tid == 0) invstd_row[row] = inv_std;
    __syncthreads();

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float xh = (x[offset + i] - mean) * inv_std;
        xhat[offset + i] = xh;
        y[offset + i] = xh * gamma[i] + beta[i];
    }
}

// Fused residual + layernorm-with-cache:
//   x_out = x + residual;  y = LN(x_out) with xhat/invstd for backward.
__global__ void residual_layernorm_cache_fp32(
    const float* x, const float* residual,
    float* x_out, float* y, float* xhat, float* invstd_row,
    const float* gamma, const float* beta,
    int hidden_dim, float eps, int total_rows
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int offset = row * hidden_dim;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = x[offset + i] + residual[offset + i];
        x_out[offset + i] = v;
        local_sum += v;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / hidden_dim;
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = x_out[offset + i] - mean;
        local_var_sum += diff * diff;
    }
    sdata[tid] = local_var_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / hidden_dim;
    float inv_std = rsqrtf(variance + eps);
    if (tid == 0) invstd_row[row] = inv_std;
    __syncthreads();

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float xh = (x_out[offset + i] - mean) * inv_std;
        xhat[offset + i] = xh;
        y[offset + i] = xh * gamma[i] + beta[i];
    }
}

// a[i] += b[i] (non-atomic; a and b must not alias overlapping warps incorrectly — same shape OK)
__global__ void add_into_fp32(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Layernorm backward: one block per row. Writes dx; dgamma/dbeta via atomicAdd.
__global__ void layernorm_backward_fp32(
    const float* dout, const float* xhat, const float* invstd_row, const float* gamma,
    float* dx, float* dgamma, float* dbeta,
    int hidden_dim, int total_rows
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    int offset = row * hidden_dim;
    float invstd = invstd_row[row];
    float N = (float)hidden_dim;

    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float d = dout[offset + i];
        float xh = xhat[offset + i];
        float dxh = d * gamma[i];
        sum_dxhat += dxh;
        sum_dxhat_xhat += dxh * xh;
        atomicAdd(&dgamma[i], d * xh);
        atomicAdd(&dbeta[i], d);
    }
    sdata[tid] = sum_dxhat;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum_dxhat = sdata[0];
    __syncthreads();

    sdata[tid] = sum_dxhat_xhat;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum_dxhat_xhat = sdata[0];
    __syncthreads();

    float scale = invstd / N;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float xh = xhat[offset + i];
        float dxh = dout[offset + i] * gamma[i];
        dx[offset + i] = scale * (N * dxh - row_sum_dxhat - xh * row_sum_dxhat_xhat);
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

// Elementwise add: out = a + b (same shape)
__global__ void add_fp32(const float* a, const float* b, float* out, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        out[idx] = a[idx] + b[idx];
    }
}

// Warp-reduction helper (Kepler sm_35: __shfl_down, not __shfl_down_sync).
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused causal multi-head attention for one (batch, head, query_row).
// Q,K,V layout: [B*T, C] row-major, C = H*hd; head h cols [h*hd : (h+1)*hd].
// blockDim.x must equal hd.
__global__ void causal_mha_fp32(
    const float* Q, const float* K, const float* V,
    float* O, float* Probs,
    int B, int T, int H, int hd, float scale
) {
    int row = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;

    if (row >= T || head >= H || batch >= B || tid >= hd) return;

    int C = H * hd;
    extern __shared__ float smem[];
    float* s_scores = smem;                    // [T]
    float* s_probs = smem + T;                 // [T]
    float* q_sh = smem + 2 * T;                // [hd]
    float* warp_sums = smem + 2 * T + hd;      // [ceil(hd/32)]

    const float* q_row = Q + (batch * T + row) * C + head * hd;
    float* o_row = O + (batch * T + row) * C + head * hd;
    int prob_row_base = ((batch * H + head) * T + row) * T;
    int num_warps = (hd + 31) / 32;

    q_sh[tid] = q_row[tid];
    __syncthreads();

    for (int j = 0; j <= row; ++j) {
        const float* k_row = K + (batch * T + j) * C + head * hd;
        float partial = q_sh[tid] * k_row[tid];
        float warp_sum = warp_reduce_sum(partial);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = warp_sum;
        __syncthreads();
        if (tid == 0) {
            float dot = 0.0f;
            for (int w = 0; w < num_warps; ++w) dot += warp_sums[w];
            s_scores[j] = dot * scale;
        }
        // Required: prevent other warps from overwriting warp_sums while tid0 reduces.
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = -1e30f;
        for (int j = 0; j <= row; ++j) max_val = fmaxf(max_val, s_scores[j]);
        float sum = 0.0f;
        for (int j = 0; j <= row; ++j) {
            float e = expf(s_scores[j] - max_val);
            s_probs[j] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j <= row; ++j) {
            s_probs[j] *= inv_sum;
            Probs[prob_row_base + j] = s_probs[j];
        }
        for (int j = row + 1; j < T; ++j) Probs[prob_row_base + j] = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j <= row; ++j) {
        const float* v_row = V + (batch * T + j) * C + head * hd;
        acc += s_probs[j] * v_row[tid];
    }
    o_row[tid] = acc;
}

// Split [rows, 3*C] qkv into contiguous Q,K,V [rows, C] (avoids strided views).
__global__ void split_qkv_fp32(const float* qkv, float* q, float* k, float* v, int rows, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * C;
    if (idx >= n) return;
    int row = idx / C;
    int col = idx % C;
    int base = row * 3 * C;
    q[idx] = qkv[base + col];
    k[idx] = qkv[base + C + col];
    v[idx] = qkv[base + 2 * C + col];
}

// Cross-entropy: one block per row. Writes probs into d_logits workspace, then
// subtracts 1 at target index and scales by 1/rows. loss_out[0] accumulates -log(p[target]).
__global__ void cross_entropy_fp32(
    const float* logits, const int* targets, float* d_logits, float* loss_out,
    int vocab_size, int total_rows
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    int offset = row * vocab_size;
    int target = targets[row];

    float local_max = -1e9f;
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

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float e = expf(logits[offset + i] - row_max);
        d_logits[offset + i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float inv = 1.0f / row_sum;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        d_logits[offset + i] *= inv;
    }
    __syncthreads();
    if (tid == 0) {
        float correct_p = d_logits[offset + target];
        atomicAdd(loss_out, -logf(fmaxf(correct_p, 1e-12f)));
        d_logits[offset + target] -= 1.0f;
    }
    __syncthreads();
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        d_logits[offset + i] /= (float)total_rows;
    }
}

// GeLU backward (tanh approximation) elementwise.
__global__ void gelu_backward_fp32(const float* x, const float* d_out, float* d_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = x[i];
    float k = 0.79788456f;
    float c = 0.044715f;
    float inner = k * (xv + c * xv * xv * xv);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float g = 0.5f * (1.0f + t) + 0.5f * xv * sech2 * k * (1.0f + 3.0f * c * xv * xv);
    d_x[i] = d_out[i] * g;
}

// scale array in place
__global__ void scal_mul_fp32(float* data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

// AdamW update for one flattened parameter tensor
__global__ void adamw_update_fp32(
    float* w, const float* g, float* m, float* v,
    float lr, float wd, float b1, float b2, float eps,
    float bc1, float bc2, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = g[i];
    float mm = b1 * m[i] + (1.0f - b1) * grad;
    float vv = b2 * v[i] + (1.0f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    float m_hat = mm / bc1;
    float v_hat = vv / bc2;
    w[i] -= lr * (m_hat / (sqrtf(v_hat) + eps));
    if (wd > 0.0f) w[i] -= lr * wd * w[i];
}

// Token embedding backward: atomicAdd into embedding table rows.
__global__ void embed_backward_fp32(
    float* emb_grad, const int* ids, const float* d_h, int B, int T, int C
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    if (b >= B || t >= T || tid >= C) return;
    int id = ids[b * T + t];
    atomicAdd(&emb_grad[id * C + tid], d_h[(b * T + t) * C + tid]);
}

// Position embedding backward: sum d_h over batch for each (t, c).
__global__ void pos_embed_backward_fp32(
    float* pos_grad, const float* d_h, int B, int T, int C
) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= T || tid >= C) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += d_h[(b * T + t) * C + tid];
    }
    atomicAdd(&pos_grad[t * C + tid], acc);
}

// Fused softmax backward for attention: d_scores = probs * (d_probs - row_dot) * scale
// probs/d_probs/d_scores are [total_rows, T] row-major (one row per query position).
__global__ void softmax_backward_fp32(
    const float* probs,
    const float* d_probs,
    float* d_scores,
    int T,
    int total_rows,
    float scale
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int offset = row * T;

    float local_dot = 0.0f;
    for (int i = tid; i < T; i += blockDim.x) {
        local_dot += probs[offset + i] * d_probs[offset + i];
    }
    sdata[tid] = local_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_dot = sdata[0];
    __syncthreads();

    for (int i = tid; i < T; i += blockDim.x) {
        d_scores[offset + i] = probs[offset + i] * (d_probs[offset + i] - row_dot) * scale;
    }
}

// Add block [block_rows, hd] into acc[(row0+r)*C + col_start + c].
__global__ void add_block_fp32(
    float* acc, const float* block,
    int row0, int C, int col_start, int hd, int block_rows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = block_rows * hd;
    if (idx >= n) return;
    int r = idx / hd;
    int c = idx % hd;
    atomicAdd(&acc[(row0 + r) * C + col_start + c], block[r * hd + c]);
}

// Accumulate squared gradient elements into out[0] (for global norm).
__global__ void grad_norm_contrib_fp32(const float* g, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, g[i] * g[i]);
}

// Elementwise add: out += block (same shape, n elements).
__global__ void add_inplace_fp32(float* out, const float* block, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&out[i], block[i]);
}

// Batched tiled GEMM with optional transpose: C[b]=op(A[b]) @ op(B[b]).
// A,B,C are contiguous [batch, ...] row-major.
// If transA=0, A[b] is [M,K]; if transA=1, A[b] is stored [K,M] and read as A^T -> [M,K].
// If transB=0, B[b] is [K,N]; if transB=1, B[b] is stored [N,K] and read as B^T -> [K,N].
// C[b] is always [M,N]. Uses GEMM_TILE shared-memory tiles (Kepler-friendly).
__global__ void gemm_batched_fp32(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int batch,
    int transA, int transB
) {
    __shared__ float sA[GEMM_TILE][GEMM_TILE];
    __shared__ float sB[GEMM_TILE][GEMM_TILE];

    int bat = blockIdx.z;
    if (bat >= batch) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE + ty;
    int col = blockIdx.x * GEMM_TILE + tx;

    const float* Ab = A + bat * (M * K);
    const float* Bb = B + bat * (K * N);
    float* Cb = C + bat * (M * N);

    float sum = 0.0f;
    int tiles = (K + GEMM_TILE - 1) / GEMM_TILE;

    for (int t = 0; t < tiles; ++t) {
        int a_k = t * GEMM_TILE + tx;
        int b_k = t * GEMM_TILE + ty;

        if (row < M && a_k < K) {
            sA[ty][tx] = transA ? Ab[a_k * M + row] : Ab[row * K + a_k];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (b_k < K && col < N) {
            sB[ty][tx] = transB ? Bb[col * K + b_k] : Bb[b_k * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < GEMM_TILE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = sum;
    }
}

// Token + position embedding forward: out[(b*T+t)*C+c] = tok_emb[id,c] + pos_emb[t,c]
__global__ void embed_forward_fp32(
    const float* tok_emb, const float* pos_emb, const int* ids,
    float* out, int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * C;
    if (idx >= n) return;
    int c = idx % C;
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int id = ids[b * T + t];
    out[idx] = tok_emb[id * C + c] + pos_emb[t * C + c];
}

// ============================================================================
// MHA backward suite (ported from llm gpu 5/core/mha_kernels.py)
// Score-Space:   [H, M, M]  with H=B*NH, M=T
// Projection:    [H, M, D]  with D=HD
// Interleaved:   [B*T, NH*HD]
// ============================================================================

// [B*T, C] head-interleaved -> [B*NH, T, HD]
__global__ void interleaved_to_heads(
    const float* X, float* Out,
    int B, int T, int NH, int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int in_idx = (b * T + t) * (NH * HD) + nh * HD + hd;
    int out_idx = (b * NH + nh) * T * HD + t * HD + hd;
    Out[out_idx] = X[in_idx];
}

// [B*NH, T, HD] -> [B*T, C]
__global__ void merge_heads_kernel(
    const float* ContextHeads, float* Context,
    int B, int T, int NH, int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int in_idx = (b * NH + nh) * T * HD + t * HD + hd;
    int out_idx = (b * T + t) * (NH * HD) + nh * HD + hd;
    Context[out_idx] = ContextHeads[in_idx];
}

// Pack Q,K,V [rows, C] into qkv [rows, 3*C]
__global__ void pack_qkv_fp32(
    const float* q, const float* k, const float* v, float* qkv,
    int rows, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * C;
    if (idx >= n) return;
    int row = idx / C;
    int col = idx % C;
    int base = row * 3 * C;
    qkv[base + col] = q[idx];
    qkv[base + C + col] = k[idx];
    qkv[base + 2 * C + col] = v[idx];
}

// Scores[h,i,j] = sum_d A[h,i,d] * B[h,j,d] * scale  (used for dProbs = dOut @ V^T)
__global__ void matmul_score_kernel(
    const float* A, const float* B, float* Scores,
    int H, int M, int D, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * M;
    if (idx >= total) return;

    int h = idx / (M * M);
    int rem = idx - h * (M * M);
    int i = rem / M;
    int j = rem - i * M;

    int a_base = h * M * D + i * D;
    int b_base = h * M * D + j * D;

    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        acc += A[a_base + d] * B[b_base + d];
    }
    Scores[h * M * M + i * M + j] = acc * scale;
}

// Softmax VJP: dRaw = scale * probs * (dScores - dot); causal j<=i. Scale fused (was separate scal_mul).
__global__ void softmax_fused_backward(
    const float* dScores, const float* probs,
    float* row_sum, float* dProbs,
    int H, int M, float scale
) {
    extern __shared__ float shared_buf[];

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= H || i >= M) return;

    int row_base = h * M * M + i * M;
    int valid_cols = i + 1;

    float thread_dot = 0.0f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        thread_dot += dScores[row_base + j] * probs[row_base + j];
    }
    shared_buf[tid] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
        __syncthreads();
    }
    float dot_val = shared_buf[0];
    __syncthreads();

    if (tid == 0) {
        row_sum[h * M + i] = dot_val;
    }

    for (int j = tid; j < valid_cols; j += blockDim.x) {
        dProbs[row_base + j] = scale * probs[row_base + j] * (dScores[row_base + j] - dot_val);
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        dProbs[row_base + j] = 0.0f;
    }
}

// dQ = dRaw @ K
__global__ void matmul_grad_q_kernel(
    const float* dProbs, const float* K, float* dQ,
    int H, int M, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int i = rem / D;
    int d = rem - i * D;

    int row_base = h * M * M + i * M;
    int k_col_base = h * M * D + d;

    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        acc += dProbs[row_base + j] * K[k_col_base + j * D];
    }
    dQ[h * M * D + i * D + d] = acc;
}

// dK = dRaw^T @ Q
__global__ void matmul_grad_k_kernel(
    const float* dProbs, const float* Q, float* dK,
    int H, int M, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int j = rem / D;
    int d = rem - j * D;

    int q_col_base = h * M * D + d;
    int scores_base = h * M * M;

    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
        acc += dProbs[scores_base + i * M + j] * Q[q_col_base + i * D];
    }
    dK[h * M * D + j * D + d] = acc;
}

// dV = probs^T @ dOut
__global__ void matmul_grad_v(
    const float* probs, const float* dOut, float* dV,
    int H, int M, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int j = rem / D;
    int d = rem - j * D;

    int dout_col_base = h * M * D + d;
    int probs_base = h * M * M;

    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
        acc += probs[probs_base + i * M + j] * dOut[dout_col_base + i * D];
    }
    dV[h * M * D + j * D + d] = acc;
}

// Transpose 2D matrix: out[c, r] = in[r, c]. in is [rows, cols], out is [cols, rows].
__global__ void transpose_2d_fp32(const float* in, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) return;
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[idx];
}

// Column sum: output[c] = sum_r input[r, c].
// One block per channel; threads cooperatively reduce over rows.
__global__ void reduce_sum_axis0_fp32(
    const float* input, float* output, int num_rows, int channels
) {
    extern __shared__ float sdata[];
    int c = blockIdx.x;
    if (c >= channels) return;
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int r = tid; r < num_rows; r += blockDim.x) {
        acc += input[r * channels + c];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[c] = sdata[0];
}

// QKV [B*T, 3*C] -> Q/K/V [B*NH, T, HD]
__global__ void split_heads_kernel(
    const float* QKV, float* Q, float* K, float* V,
    int B, int T, int NH, int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = NH * HD;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int qkv_row = b * T + t;
    int col = nh * HD + hd;
    int out_idx = (b * NH + nh) * T * HD + t * HD + hd;

    Q[out_idx] = QKV[qkv_row * 3 * C + col];
    K[out_idx] = QKV[qkv_row * 3 * C + C + col];
    V[out_idx] = QKV[qkv_row * 3 * C + 2 * C + col];
}

// Pack dQ/dK/dV [B*NH, T, HD] into dQKV [B*T, 3*C]
__global__ void merge_heads_qkv_kernel(
    const float* dQ, const float* dK, const float* dV, float* dQKV,
    int B, int T, int NH, int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = NH * HD;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int in_idx = (b * NH + nh) * T * HD + t * HD + hd;
    int qkv_row = b * T + t;
    int col = nh * HD + hd;
    dQKV[qkv_row * 3 * C + col] = dQ[in_idx];
    dQKV[qkv_row * 3 * C + C + col] = dK[in_idx];
    dQKV[qkv_row * 3 * C + 2 * C + col] = dV[in_idx];
}

// Phase 2C: fused QKt + causal softmax + PV. One block per (h, i) row.
// Q/K/V/Out: [H, M, D]; Scores: [H, M, M] normalized probs.
__global__ void fused_attention_forward_kernel(
    const float* Q, const float* K, const float* V,
    float* Scores, float* Out,
    float* row_max, float* row_sum,
    int H, int M, int D, float scale
) {
    extern __shared__ float smem[];
    float* q_row = smem;
    float* row_scores = smem + D;
    float* scratch = smem + D + M;

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= H || i >= M) return;

    int valid_cols = i + 1;
    int q_base = h * M * D + i * D;

    for (int d = tid; d < D; d += blockDim.x) {
        q_row[d] = Q[q_base + d];
    }
    __syncthreads();

    for (int j = tid; j < valid_cols; j += blockDim.x) {
        int k_base = h * M * D + j * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q_row[d] * K[k_base + d];
        }
        row_scores[j] = acc * scale;
    }
    __syncthreads();

    float thread_max = -1e30f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        thread_max = fmaxf(thread_max, row_scores[j]);
    }
    scratch[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = scratch[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float exp_v = expf(row_scores[j] - max_val);
        row_scores[j] = exp_v;
        thread_sum += exp_v;
    }
    scratch[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = scratch[0];
    __syncthreads();

    if (tid == 0) {
        row_max[h * M + i] = max_val;
        row_sum[h * M + i] = sum_val;
    }

    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        row_scores[j] *= inv_sum;
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        row_scores[j] = 0.0f;
    }
    __syncthreads();

    int row_base = h * M * M + i * M;
    for (int j = tid; j < M; j += blockDim.x) {
        Scores[row_base + j] = row_scores[j];
    }
    __syncthreads();

    // PV: parallelize over D (avoid serial D loop + sync per channel).
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < valid_cols; ++j) {
            acc += row_scores[j] * V[h * M * D + j * D + d];
        }
        Out[h * M * D + i * D + d] = acc;
    }
}
"""
