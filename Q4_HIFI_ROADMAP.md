## 🗺️ **Q4_HIFI Roadmap: Adaptive Quantization for All Model Sizes**

### 🎯 **Mission Statement** 
> **Build an intelligent 4.3–4.5 BPW quantization that automatically optimizes for any model size (0.5B–200B+), beating Q4_K_M in quality, size, and speed through parameter-driven, adaptive outlier preservation.**

---

## 🔬 **Phase 1: Intelligent Baseline (2–3 days)** 
**Objective**: Implement **automatic model detection + adaptive quantization** that works out-of-the-box for any GGUF model.

### ✅ **Core Components**
1. **Parameter Counting (Primary Decision Point)** 
   - Sum all tensor element counts from GGUF metadata
   - Use parameter count directly for outlier scaling — no architecture lookup needed
   - Works automatically for any model, known or unknown

2. **Parameter-Based Outlier Scaling** 
   ```c
   // Universal parameter-based outlier selection
   static int get_base_outliers(int64_t param_count) {
       if (param_count <= 3e9)  return 8;   // ≤3B:    8 outliers
       if (param_count <= 30e9) return 10;  // 3B-30B: 10 outliers
       if (param_count <= 70e9) return 12;  // 30B-70B: 12 outliers
       return 16;                            // >70B:   16 outliers
   }
   
   // Massive layers always get 2× base outliers
   static int get_massive_outliers(int64_t param_count) {
       return get_base_outliers(param_count) * 2;
   }
   ```

3. **Tensor Classification (Architecture-Agnostic)** 
   - **Massive layers**: Pattern match `lm_head`, `token_embd`, `output` → 2× outliers
   - **Standard layers**: All other weight tensors → base outliers
   - **Skip**: Biases, norms, embeddings (other than token_embd)

4. **Optional Architecture Hints** 
   - Extract `general.architecture` for logging/diagnostics only
   - NOT used for outlier decisions — parameter count is sufficient
   - Useful for identifying MoE models (may want extra outliers on `ffn_gate`)

### ✅ **Implementation Steps**
1. Add `gguf_get_total_parameter_count()` to count all tensor elements
2. Implement `get_base_outliers()` and `get_massive_outliers()` using parameter thresholds
3. Add `is_massive_tensor()` pattern matcher for `lm_head`, `token_embd`, `output`
4. Create `quantize_state` struct with `param_count`, `base_outliers`, `massive_outliers`
5. Define `Q4_HIFI` block with **32 outlier slots** (fixed size for GPU compatibility, supports up to >70B models)
6. Implement quantization that selects outlier count per-tensor based on tensor name

### 📈 **Success Criteria**
- ✅ **Automatic scaling**: Any >70B model → 16 base, 32 massive outliers
- ✅ **Automatic scaling**: Any ≤3B model → 8 base, 16 massive outliers
- ✅ **Architecture-agnostic**: Works identically for Llama, Qwen, Mistral, DeepSeek, etc.
- ✅ **Zero manual configuration** required — parameter count drives everything
- ✅ **Baseline PPL** ≤ Q4_K_M on all tested models
- ✅ **Future-proof**: New architectures work automatically without code changes

---

## ⚡ **Phase 2: GPU-Optimized Performance (2–3 days)** 
**Objective**: Achieve **≥95% of Q4_K_M speed** on GPU while maintaining quality advantage.

### ✅ **Core Components**
1. **GPU Transcoding Kernel** 
   - Convert `Q4_HIFI` → `Q4_K` layout on GPU load 
   - Reuse battle-tested `Q4_K_CUDA` kernel

2. **Outlier Correction Kernel** 
   - Minimal overhead: 16–32 outliers per block 
   - Coalesced memory access for massive layers

3. **CPU SIMD Optimization** 
   - `vec_dot_q4_hifi_q8_K_avx2` with Q4_K bulk + outlier loop 
   - Outlier skipping for `|q8_val| < threshold`

### ✅ **Implementation Steps**
1. Add `k_q4_hifi_to_gpu` transcoding kernel to `ggml-cuda.cu`
2. Implement `k_add_outlier_correction_q4_hifi` with adaptive count
3. Integrate into matmul pipeline with proper memory management
4. Add CPU AVX2 kernel with outlier skipping optimization

### 📈 **Target Metrics (Distrill-123B)**
| Metric | Q4_K_M | Q4_HIFI Target |
|--------|--------|----------------|
| **GPU Speed** | 817 tok/s | **≥780 tok/s** |
| **CPU Speed** | 120 tok/s | **≥110 tok/s** |
| **VRAM Overhead** | 0% | **≤2%** |

---

## 🧠 **Phase 3: Large-Model Specialization (2 days)** 
**Objective**: Maximize quality gain on **70B–123B models** through domain-optimized strategies.

### ✅ **Core Components**
1. **Domain-Mixed Imatrix** 
   - 40% Wikitext, 30% Code, 30% Math for 123B models 
   - Automatic dataset selection based on architecture

2. **Massive Layer Optimization** 
   - **32 outliers for `lm_head`/`token_embd`** (128K vocab × 123B params) 
   - **Pre-zeroing** in Q8_K activations for faster outlier correction

3. **Hybrid Tensor Strategy** 
   ```bash
   --tensor-type lm_head=Q4_HIFI      # 32 outliers
   --tensor-type token_embd=Q4_HIFI   # 32 outliers 
   --tensor-type ffn_gate=Q4_HIFI     # 16 outliers (MoE critical)
   --tensor-type ffn_down=Q4_HIFI     # 16 outliers
   --tensor-type attn_v=Q4_HIFI       # 16 outliers
   --tensor-type ffn_up=Q4_HIFI       # 16 outliers
   ```

### ✅ **Implementation Steps**
1. Create domain-mixed imatrix generation script
2. Enhance `get_tensor_outlier_count()` for architecture-specific rules
3. Implement pre-zeroing in Q8_K quantization for `Q4_HIFI` tensors
4. Test on Distrill-123B with HumanEval + MATH benchmarks

### 📈 **Target Metrics (Distrill-123B)**
| Metric | Q4_K_M | Q4_HIFI Target |
|--------|--------|----------------|
| **Wikitext PPL** | 11.94 | **≤11.6** |
| **HumanEval pass@1** | 35.2% | **≥38.0%** |
| **File Size** | 60.2 GiB | **≤58.5 GiB** |

---

## 📊 **Expected Outcomes by Parameter Count**

*Outlier counts are determined solely by parameter count — architecture is irrelevant.*

| Parameter Count | Base Outliers | Massive Outliers | Expected PPL Gain |
|-----------------|---------------|------------------|-------------------|
| **≤3B** | 8 | 16 | +0.1–0.2 |
| **3B–30B** | 10 | 20 | +0.2–0.3 |
| **30B–70B** | 12 | 24 | +0.3–0.4 |
| **>70B** | **16** | **32** | **+0.4–0.6** |

---

## 🔧 **Technical Implementation Details**

### **Block Format (Fixed Size for GPU)**
```c
typedef struct {
    uint8_t qs[128];           // Q4_K core (128 bytes)
    uint8_t scales[32];        // Q4_K scales (32 bytes) 
    ggml_fp16_t d;             // Q4_K scale (2 bytes)
    uint8_t outlier_count;     // Actual count: 8-32 depending on model size (1 byte)
    uint8_t outlier_idx[32];   // Padded to max for >70B massive layers (32 bytes)
    ggml_fp16_t outlier_vals[32]; // Padded to max (64 bytes)
} block_q4_hifi; // Total: 259 bytes max (but outlier slots often unused)
// Effective BPW: 4.3-4.5 (actual outliers << max slots for most layers)
```

### **Automatic Workflow**
```bash
# User runs single command - everything auto-detected
./llama-quantize model-f16.gguf model-q4hifi.gguf Q4_HIFI

# System automatically:
# 1. Counts total parameters from GGUF tensor metadata
# 2. Selects outlier counts based on parameter thresholds:
#    - 123B model → 16 base, 32 massive (>70B tier)
#    - 7B model   → 10 base, 20 massive (3B-30B tier)
#    - 1B model   →  8 base, 16 massive (≤3B tier)
# 3. Identifies massive tensors by name pattern (lm_head, token_embd, output)
# 4. Applies appropriate outlier count per-tensor
# 5. Uses domain-mixed imatrix if available
```

---

## 🚀 **Implementation Priority**

| Priority | Phase | Why |
|----------|-------|-----|
| 🔥 **1** | **Phase 1 (Intelligent Baseline)** | Foundation for everything else |
| ⚡ **2** | **Phase 2 (GPU Performance)** | Speed is non-negotiable for 123B |
| 🧠 **3** | **Phase 3 (Large-Model Specialization)** | Competitive advantage on massive models |

---

## 💡 **Key Innovations**

1. **Parameter-driven quantization** — no architecture tables to maintain, works for any model
2. **Adaptive outlier allocation** scales automatically from 0.5B to 200B+ models 
3. **Massive layer specialization** addresses 128K vocab bottleneck with 2× outliers
4. **Domain-mixed imatrix** ensures outliers matter for math/code tasks 
5. **GPU transcoding** maintains Q4_K_M speed while adding outlier precision
6. **Future-proof design** — new architectures work immediately without code changes

---

## 🏁 **Success Definition**

**Q4_HIFI is ready when**: 
✅ **123B class models**: PPL ≤11.6, Size ≤58.5 GiB, Speed ≥780 tok/s 
✅ **Parameter-based detection** works for any GGUF model without architecture-specific code
✅ **Zero configuration** required — users just specify `Q4_HIFI` 
✅ **Beats Q4_K_M** on quality/size/speed across all model sizes
✅ **Future-proof** — new model architectures work automatically

This roadmap delivers **the first truly intelligent quantization format** that **adapts to your model based on parameter count, not hardcoded architecture tables**.