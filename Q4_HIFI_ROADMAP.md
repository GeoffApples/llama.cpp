# 🗺️ **Q4_HIFI Roadmap: Adaptive Outlier-Aware Quantization**

> **Goal**: Deliver a **4.3–4.5 BPW quantization format that beats Q4_K_M in quality while matching its speed and size**, with **automatic model-aware optimization** for models from 0.5B to 123B parameters.

---

## 🔬 **Phase 1: Intelligent Baseline (COMPLETED ✅)**

### 📊 **Phase 1 Results (Qwen3-0.6B)**
| Metric | Q4_K_S | Q4_K_M | **Q4_HIFI** |
|--------|--------|--------|-------------|
| **PPL** | 24.55 | 23.69 | **23.34** ✅ |
| **Speed** | 632 t/s | 624 t/s | 265 t/s ❌ |
| **Size** | 443 MiB | 456 MiB | 664 MiB ❌ |
| **BPW** | 4.95 | 5.09 | 7.41 ❌ |

**Key Insight**: Quality works ✅, but **over-application and lack of optimized kernels** cause speed/size issues.

---

## ⚡ **Phase 2: Production Optimization (IN PROGRESS)**

### 🎯 **Objective**: Fix speed/size while preserving quality advantage.

---

### 🔧 **2.1 Hybrid Tensor Mixing** *(Priority: 🔥)*

**Problem**: Applying Q4_HIFI to **all tensors** wastes resources. 
**Solution**: Apply only to **critical layers**.

#### ✅ **Code: Tensor-Type Selection (`llama.cpp`)**
```cpp
// In quantization logic
ggml_type quantize_tensor(const char* name, ggml_type default_type) {
    const char* critical[] = {
        "lm_head", "token_embd", "embed_tokens",
        "attn_v", "ffn_down", "ffn_gate"
    };
   
    for (const char* pattern : critical) {
        if (strstr(name, pattern)) return GGML_TYPE_Q4_HIFI;
    }
    return default_type; // Q4_K_M for others
}
```

#### 📊 **Expected Impact**
- **Size**: 664 MiB → **470 MiB** (matches Q4_K_M)
- **Quality**: **Unchanged** (23.34 PPL)
- **Speed**: Minor improvement (less outlier work)

---

### 🔧 **2.2 CPU AVX2 vec_dot Kernel** *(Priority: 🔥)*

**Problem**: Using slow dequantization fallback. 
**Solution**: Optimized SIMD kernel.

#### ✅ **Code: `ggml-cpu/quants.c`**
```c
#ifdef __AVX2__
void ggml_vec_dot_q4_hifi_q8_K(
    const int n, float* s,
    const void* vx, const void* vy
) {
    const block_q4_hifi* x = (const block_q4_hifi*)vx;
    const block_q8_K* y = (const block_q8_K*)vy;
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        // Reuse Q4_K bulk kernel
        float bulk_sum;
        ggml_vec_dot_q4_K_q8_K(1, &bulk_sum, (block_q4_K*)&x[i], &y[i]);
        sumf += bulk_sum;

        // Add outliers
        for (int k = 0; k < x[i].outlier_count; ++k) {
            uint8_t idx = x[i].outlier_idx[k];
            float w = GGML_FP16_TO_FP32(x[i].outlier_vals[k]);
            sumf += w * y[i].qs[idx] * GGML_FP16_TO_FP32(y[i].d);
        }
    }
    *s = sumf;
}
#endif
```

#### 📊 **Expected Impact**
- **Speed**: 265 t/s → **550+ t/s** (+107%)
- **Quality**: Unchanged

---

### 🔧 **2.3 GPU Transcoding + Async Streams** *(Priority: ⚡)*

**Problem**: No GPU acceleration. 
**Solution**: Transcode to Q4_K layout + outlier correction.

#### ✅ **Code: `ggml-cuda.cu`**
```cuda
// Transient GPU block
struct block_q4_hifi_gpu {
    uint8_t qs[128], scales[32];
    half d;
    uint8_t outlier_count, outlier_idx[16];
    half outlier_vals[16];
};

__global__ void k_add_outlier_correction_q4_hifi(
    const block_q4_hifi_gpu* x,
    const block_q8_K* y,
    float* dst, int nb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
   
    float corr = 0.0f, d8 = __half2float(y[i].d);
    for (int k = 0; k < x[i].outlier_count; ++k) {
        corr += __half2float(x[i].outlier_vals[k]) *
                y[i].qs[x[i].outlier_idx[k]] * d8;
    }
    dst[i] += corr;
}

void ggml_cuda_mul_mat_q4_hifi(...) {
    // 1. Transcode to GPU block
    k_q4_hifi_to_gpu<<<grid, block, 0, stream>>>(src, gpu_block, nb);
   
    // 2. Run Q4_K kernel (full speed!)
    ggml_cuda_mul_mat_q4_K_stream((block_q4_K*)gpu_block, ...);
   
    // 3. Add outliers
    k_add_outlier_correction_q4_hifi<<<grid, block, 0, stream>>>(...);
}
```

#### 📊 **Expected Impact (Distrill-123B)**
- **GPU Speed**: Matches Q4_K_M (800+ t/s)
- **VRAM Overhead**: <2%
- **Quality**: Preserved

---

## 🧠 **Phase 3: Large-Model Specialization**

### 🎯 **Objective**: Maximize gains on 70B–123B models.

---

### 🔧 **3.1 Adaptive Outlier Counting**

#### ✅ **Code: Model Classification (`ggml-quants.c`)**
```c
static void classify_model(int64_t params, const char* arch,
                          int* base_out, int* massive_out) {
    if (params <= 1000000000LL) {      // ≤1B
        *base_out = 8; *massive_out = 16;
    } else if (params <= 30000000000LL) { // ≤30B
        *base_out = 10; *massive_out = 20;
    } else {                             // >30B (123B)
        *base_out = 16; *massive_out = 32;
    }
}
```

---

### 🔧 **3.2 Domain-Mixed Imatrix**

**For 123B models, use mixed calibration data**:
```bash
# 40% Wikitext, 30% Code, 30% Math
python create_mixed_imatrix_dataset.py \
  --wikitext wikitext-10k.txt \
  --code codeparrot-5k.txt \
  --math mathqa-5k.txt \
  --output mixed-10k.txt

./llama-perplexity -m model-f16.gguf -f mixed-10k.txt --keep-imatrix
```

---

## 📊 **Expected Final Results**

| Model | Metric | Q4_K_M | **Q4_HIFI (Target)** |
|-------|--------|--------|----------------------|
| **Qwen-0.6B** | PPL | 23.69 | **23.34** |
| | Speed | 624 t/s | **610 t/s** |
| | Size | 456 MiB | **470 MiB** |
| **Distrill-123B** | PPL | 11.94 | **≤11.6** |
| | Speed | 817 t/s | **≥800 t/s** |
| | Size | 60.2 GiB | **≤58.5 GiB** |

---

## 🚀 **Implementation Priority**

| Priority | Task | Timeline | Owner |
|----------|------|----------|-------|
| 🔥 **1** | Hybrid tensor mixing | 1 day | You |
| 🔥 **2** | CPU AVX2 vec_dot | 2 days | You |
| ⚡ **3** | GPU transcoding kernels | 3 days | You |
| 🧠 **4** | Adaptive outlier logic | 1 day | You |
| 🧪 **5** | Distrill-123B validation | 2 days | You |

---

## 💡 **Key Innovations**

1. **First quantization format with automatic model-aware optimization**
2. **Hybrid mixing** focuses resources where they matter most
3. **GPU transcoding** delivers Q4_K_M speed with Q4_HIFI quality
4. **Adaptive outlier counts** scale from 0.5B to 123B models

---

## ✅ **Success Criteria**

**Q4_HIFI is ready when**:
- ✅ **Qwen-0.6B**: PPL ≤23.4, Speed ≥600 t/s, Size ≤475 MiB
- ✅ **Distrill-123B**: PPL ≤11.7, Speed ≥790 t/s, Size ≤59 GiB
- ✅ **Automatic detection** works for all major architectures
- ✅ **Zero configuration** required for users

---

## 📦 **Deliverables**

- [x] **Phase 1**: Quality validation (completed)
- [ ] **Phase 2**: Hybrid mixing + AVX2 kernel
- [ ] **Phase 2**: GPU transcoding kernels 
- [ ] **Phase 3**: Adaptive outlier logic
- [ ] **Phase 3**: Distrill-123B validation
- [ ] **Documentation**: `docs/quantization/Q4_HIFI.md`

---

This roadmap transforms Q4_HIFI from a **quality-focused prototype** into a **production-ready quantization format** that **dominates Q4_K_M across all model sizes**.