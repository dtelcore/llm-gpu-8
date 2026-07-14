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

// Fused causal multi-head attention for one (batch, head, query_row).
// Q,K,V layout: [B*T, C] row-major, C = H*hd; head h cols [h*hd : (h+1)*hd].
// O: [B*T, C] concatenated head outputs per row.
// Probs: [B, H, T, T] flattened as ((b*H + h)*T + row)*T + col.
// blockDim.x must equal hd (head dimension); uses shared-memory reduction.
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
    float* s_scores = smem;
    float* s_probs = smem + T;
    float* red = smem + 2 * T;

    const float* q_row = Q + (batch * T + row) * C + head * hd;
    float* o_row = O + (batch * T + row) * C + head * hd;
    int prob_row_base = ((batch * H + head) * T + row) * T;

    for (int j = 0; j <= row; ++j) {
        const float* k_row = K + (batch * T + j) * C + head * hd;
        float partial = q_row[tid] * k_row[tid];
        red[tid] = partial;
        __syncthreads();

        if (tid == 0) {
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) {
                dot += red[d];
            }
            s_scores[j] = dot * scale;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = -1e9f;
        for (int j = 0; j <= row; ++j) {
            max_val = fmaxf(max_val, s_scores[j]);
        }
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
        for (int j = row + 1; j < T; ++j) {
            Probs[prob_row_base + j] = 0.0f;
        }
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
"""
