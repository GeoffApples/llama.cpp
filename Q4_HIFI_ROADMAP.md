# 🗺️ **Q4_HIFI Roadmap: Adaptive Outlier-Aware Quantization**

> **Goal**: Deliver a **4.5–5.0 BPW quantization format that beats Q4_K_M in quality while matching its speed and size**, with **automatic model-aware optimization** for models from 0.5B to 123B parameters.

---

## 🔬 **Phase 1: Intelligent Baseline (COMPLETED ✅)**

### 📊 **Phase 1 Results (Qwen3-0.6B)**
| Metric | Q4_K_S | Q4_K_M | **Q4_HIFI** |
|--------|--------|--------|-------------|
| **PPL** | 24.55 | 23.69 | **23.34** ✅ |
| **Speed** | 632 t/s | 624 t/s | 265 t/s ❌ |
| **Size** | 443 MiB | 456 MiB | 664 MiB ❌ |
| **BPW** | 4.95 | 5.09 | 7.41 ❌ |

**Root Cause Analysis**:
1. ❌ **Over-application**: ALL tensors get Q4_HIFI (should be Q4_K base + Q4_HIFI on sensitive layers)
2. ❌ **No MMVQ kernel**: Falls back to slow dequantization path on GPU
3. ❌ **No AVX2 kernel**: Generic scalar implementation on CPU

---

## ⚡ **Phase 2: Production Optimization (IN PROGRESS)**

### 🎯 **Objective**: Fix speed/size while preserving quality advantage.

---

### 🔧 **2.1 Hybrid Tensor Mixing** *(Priority: 🔥🔥🔥 CRITICAL)*

**Problem**: `default_type = GGML_TYPE_Q4_HIFI` applies Q4_HIFI to ALL 197 tensors.
**Solution**: Use Q4_K as base, apply Q4_HIFI only to **sensitive layers**.

#### 📍 **File: `src/llama-quant.cpp`**

**Change 1**: Fix default type (line ~638)
```cpp
// BEFORE (wrong - applies Q4_HIFI everywhere):
case LLAMA_FTYPE_MOSTLY_Q4_HIFI: default_type = GGML_TYPE_Q4_HIFI; break;

// AFTER (correct - Q4_K base with selective Q4_HIFI):
case LLAMA_FTYPE_MOSTLY_Q4_HIFI: default_type = GGML_TYPE_Q4_K; break;
```

**Change 2**: Add tensor selection in `llama_tensor_get_type()` (after attn_v handling)
```cpp
// For attn_v.weight tensors
else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
    // All attention value projections get Q4_HIFI (high impact on quality)
    new_type = GGML_TYPE_Q4_HIFI;
}

// For ffn_down.weight tensors  
else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
    // First half of FFN down projections (most sensitive to quantization)
    new_type = i_layer < n_layer/2 ? GGML_TYPE_Q4_HIFI : GGML_TYPE_Q4_K;
}
```

**Change 3**: Handle output/embedding tensors
```cpp
// In output tensor handling (line ~258-278)
else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
    new_type = GGML_TYPE_Q6_K; // High precision for vocabulary prediction
}
```

#### 📊 **Expected Tensor Distribution**
| Tensor Type | Count | Quantization |
|-------------|-------|--------------|
| `attn_v.weight` | 28 | Q4_HIFI |
| `ffn_down.weight` (first half) | 14 | Q4_HIFI |
| `output.weight` | 1 | Q6_K |
| All other tensors | ~155 | Q4_K |

#### 📊 **Expected Impact**
- **Size**: 664 MiB → **~470 MiB** (-29%)
- **Quality**: **Maintained** (23.34 PPL) — sensitive layers preserved
- **Speed**: Improved (fewer Q4_HIFI blocks to process)

---

### 🔧 **2.2 CUDA MMVQ Kernel** *(Priority: 🔥🔥)*

**Problem**: Q4_HIFI excluded from MMVQ path, falls back to slow cuBLAS+dequant.
**Solution**: Add dedicated `vec_dot_q4_hifi_q8_1` kernel.

#### 📍 **File: `ggml/src/ggml-cuda/vecdotq.cuh`**

```cuda
// Q4_HIFI vec_dot: Q4_K core + outlier correction
template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ float vec_dot_q4_hifi_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs
) {
    const block_q4_hifi * bq4 = (const block_q4_hifi *) vbq + kbx;
    
    // Reuse Q4_K bulk computation (first 144 bytes are identical)
    float result = vec_dot_q4_K_q8_1_impl<mmq_y, nwarps, need_check>(
        (const block_q4_K *)bq4, bq8_1, kbx, iqs);
    
    // Add outlier corrections (only thread 0 per block)
    if (iqs == 0) {
        const float d8 = __low2float(bq8_1[kbx].ds);
        #pragma unroll
        for (int k = 0; k < bq4->outlier_count && k < Q4_HIFI_MAX_OUTLIERS; ++k) {
            const int idx = bq4->outlier_idx[k];
            result += __half2float(bq4->outlier_vals[k]) * bq8_1[kbx].qs[idx] * d8;
        }
    }
    return result;
}
```

#### 📍 **File: `ggml/src/ggml-cuda/mmvq.cu`**

```cpp
// Enable MMVQ path for Q4_HIFI
case GGML_TYPE_Q4_HIFI: return vec_dot_q4_hifi_q8_1;  // Was: return nullptr

// Add switch case for mul_mat_vec_q_switch_type
case GGML_TYPE_Q4_HIFI:
    mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_HIFI>(...);
    break;
```

#### 📍 **File: `ggml/src/ggml-cuda/ggml-cuda.cu`**

```cpp
// Remove Q4_HIFI exclusion from MMVQ path (lines ~2153, ~2197)
// DELETE these lines:
&& src0->type != GGML_TYPE_Q4_HIFI;  // Q4_HIFI uses dequant path
```

#### 📊 **Expected Impact**
- **GPU Speed**: 265 t/s → **600+ t/s** (+126%)
- **Quality**: Unchanged

---

### 🔧 **2.3 CPU AVX2/NEON vec_dot Kernel** *(Priority: 🔥)*

**Problem**: Generic scalar implementation is slow.
**Solution**: SIMD-optimized kernel reusing Q4_K infrastructure.

#### 📍 **File: `ggml/src/ggml-cpu/arch/x86/quants.c`**

```c
#if defined(__AVX2__)
void ggml_vec_dot_q4_hifi_q8_K(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc
) {
    const int nb = n / QK_K;
    const block_q4_hifi * x = (const block_q4_hifi *)vx;
    const block_q8_K * y = (const block_q8_K *)vy;
    
    __m256 acc = _mm256_setzero_ps();
    
    for (int i = 0; i < nb; ++i) {
        // Process Q4_K-compatible region with AVX2 (reuse Q4_K logic)
        // ... [AVX2 Q4_K dot product implementation] ...
        
        // Scalar outlier correction (small loop, not worth vectorizing)
        float outlier_sum = 0.0f;
        const float yd = GGML_FP16_TO_FP32(y[i].d);
        for (int k = 0; k < x[i].outlier_count; ++k) {
            const int idx = x[i].outlier_idx[k];
            outlier_sum += GGML_FP16_TO_FP32(x[i].outlier_vals[k]) 
                         * y[i].qs[idx] * yd;
        }
        acc = _mm256_add_ps(acc, _mm256_set1_ps(outlier_sum));
    }
    
    *s = hsum_float_8(acc);
}
#endif
```

#### 📊 **Expected Impact**
- **CPU Speed**: Generic → **~95% of Q4_K speed**
- **Quality**: Unchanged

---

## 🧠 **Phase 3: Large-Model Specialization**

### 🎯 **Objective**: Maximize gains on 70B–123B models.

---

### 🔧 **3.1 Adaptive Outlier Counting** *(Already Implemented ✅)*

The parameter-based outlier scaling is already in place:

```cpp
// src/llama-quant.cpp (lines 83-92)
static int q4_hifi_get_base_outliers(int64_t param_count) {
    if (param_count <= 3000000000LL)  return 8;   // ≤3B:    8 outliers
    if (param_count <= 30000000000LL) return 10;  // 3B-30B: 10 outliers
    if (param_count <= 70000000000LL) return 12;  // 30B-70B: 12 outliers
    return 16;                                     // >70B:   16 outliers
}

static int q4_hifi_get_massive_outliers(int64_t param_count) {
    return q4_hifi_get_base_outliers(param_count) * 2;
}
```

---

### 🔧 **3.2 Large-Model Validation**

**Test on DeepSeek/Qwen 70B+ models**:
```bash
# Quantize with Q4_HIFI
./llama-quantize model-f16.gguf model-Q4_HIFI.gguf Q4_HIFI

# Validate perplexity
./llama-perplexity -m model-Q4_HIFI.gguf -f wikitext-2-raw/wiki.test.raw -c 512

# Benchmark speed
./llama-bench -m model-Q4_HIFI.gguf -t 8 -n 128
```

---

## 📊 **Expected Final Results**

| Model | Metric | Q4_K_M | **Q4_HIFI (Target)** |
|-------|--------|--------|----------------------|
| **Qwen-0.6B** | PPL | 23.69 | **≤23.4** |
| | Speed | 624 t/s | **≥580 t/s** (93%) |
| | Size | 456 MiB | **≤480 MiB** |
| **70B+ Models** | PPL | baseline | **≤baseline - 2%** |
| | Speed | baseline | **≥95% of baseline** |
| | Size | baseline | **≤baseline + 5%** |

---

## 🚀 **Implementation Priority**

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 🔥🔥🔥 **1** | **Hybrid tensor mixing** | Size: -29%, Speed: +10% | 1 hour |
| 🔥🔥 **2** | **CUDA MMVQ kernel** | GPU Speed: +126% | 2-3 hours |
| 🔥 **3** | **CPU AVX2 kernel** | CPU Speed: +50% | 2-3 hours |
| 🧪 **4** | **Validation & tuning** | Quality assurance | 1-2 hours |

---

## 💡 **Key Innovations**

1. **Hybrid tensor mixing** — Q4_HIFI on sensitive layers (attn_v, ffn_down), Q4_K elsewhere
2. **Parameter-driven outliers** — Automatic 8-32 outlier scaling based on model size
3. **Q4_K compatibility** — First 144 bytes match Q4_K, enabling kernel reuse
4. **Architecture-agnostic** — Works on Qwen, LLaMA, DeepSeek, etc.

---

## ✅ **Success Criteria**

**Phase 2 Complete When**:
- [ ] **Size**: ≤480 MiB for Qwen-0.6B (currently 664 MiB)
- [ ] **GPU Speed**: ≥580 t/s (currently 265 t/s)  
- [ ] **Quality**: PPL ≤23.4 maintained (currently 23.34 ✅)

**Q4_HIFI Production-Ready When**:
- [ ] Passes all above criteria
- [ ] Works on 70B+ models without regression
- [ ] Zero configuration required for users
- [ ] Documentation complete

---

## 📦 **Deliverables Checklist**

- [x] **Phase 1**: Core quantization format
- [x] **Phase 1**: CPU dequantization
- [x] **Phase 1**: CUDA dequantization  
- [x] **Phase 1**: Quality validation (PPL 23.34 ✅)
- [ ] **Phase 2.1**: Hybrid tensor mixing
- [ ] **Phase 2.2**: CUDA MMVQ kernel
- [ ] **Phase 2.3**: CPU AVX2 kernel
- [ ] **Phase 3**: Large-model validation
- [ ] **Docs**: `docs/quantization/Q4_HIFI.md`

---

## 🎯 **Next Action**

**Start with Phase 2.1 (Hybrid Tensor Mixing)** — This is the highest-impact change:
1. Modify `default_type` from `GGML_TYPE_Q4_HIFI` to `GGML_TYPE_Q4_K`
2. Add Q4_HIFI selection logic in `llama_tensor_get_type()` for sensitive layers
3. Re-quantize and validate size reduction

This single change should reduce file size from **664 MiB → ~470 MiB** while maintaining quality.