# Q3_HIFI Quantization Format

## Overview

**Q3_HIFI** is an adaptive 3-bit quantization format that combines Q3_K efficiency with improved quality through selective FP16 outlier preservation. It automatically applies Q3_HIFI to sensitive layers (attn_v, early ffn_down) while using Q3_K/Q4_K elsewhere.

## Key Features

| Feature | Value |
|---------|-------|
| Bits per weight | ~4.1 bpw (adaptive) |
| Block size | 256 weights |
| Outliers per block | 8 (FP16) |
| Block structure | Q3_K-compatible + outlier tail (134 bytes) |

## Performance Comparison

Tested on Qwen3-1.7B (CPU, 6 threads):

| Format | Size | Perplexity | Speed | Notes |
|--------|------|------------|-------|-------|
| Q3_K_S | 949 MiB | ~24 | 32.4 tok/s | Fastest, worst quality |
| **Q3_HIFI** | **993 MiB** | **18.12** | **26.9 tok/s** | **25 MiB smaller than Q3_K_M** |
| Q3_K_M | 1018 MiB | 17.69 | 27.2 tok/s | Best quality |

## Block Structure

```c
typedef struct {
    // === Q3_K-COMPATIBLE REGION (110 bytes) ===
    uint8_t hmask[32];     // 32 bytes: high bit mask (1 bit per weight)
    uint8_t qs[64];        // 64 bytes: low 2 bits (2 bits per weight)
    uint8_t scales[12];    // 12 bytes: 16 sub-group scales (6-bit each)
    ggml_half d;           // 2 bytes: super-block scale
    
    // === OUTLIER EXTENSION (24 bytes) ===
    uint8_t outlier_idx[8];    // 8 bytes: outlier positions (0-255)
    ggml_half outlier_vals[8]; // 16 bytes: FP16 outlier values
} block_q3_hifi;  // Total: 134 bytes
```

## How It Works

### Adaptive Routing
When you quantize with `Q3_HIFI`, different tensors get different treatment:
- **attn_v (all layers)**: Q3_HIFI type with 8 FP16 outliers
- **ffn_down (first 1/3)**: Q3_HIFI type with 8 FP16 outliers  
- **Other tensors**: Q3_K or Q4_K (standard k-quants)

### Quantization (for Q3_HIFI tensors)
1. Identify the 8 weights with highest magnitude × importance (from imatrix)
2. Store these outliers as exact FP16 values
3. Set outlier positions to zero in the Q3_K bulk data
4. Quantize remaining weights using standard Q3_K encoding

### Inference (vec_dot)
1. Compute Q3_K-style bulk dot product (pre-zeroed outliers contribute 0)
2. Add outlier corrections: `sum += outlier_val[k] * activation[outlier_idx[k]]`

### Why Pre-Zeroing Works
By storing zero at outlier positions during quantization, the bulk SIMD dot product naturally skips outliers. This eliminates the need for subtraction during inference.

## Usage

### Creating a Q3_HIFI Model

**Using llama-quantize (recommended):**
```bash
# Basic quantization
./llama-quantize model-f16.gguf model-q3hifi.gguf Q3_HIFI

# With importance matrix (recommended for best quality)
./llama-quantize --imatrix imatrix.gguf model-f16.gguf model-q3hifi.gguf Q3_HIFI
```

**Using Python (convert_hf_to_gguf.py):**
```bash
# Convert and quantize in one step
python convert_hf_to_gguf.py model_dir --outtype q3_hifi --outfile model-q3hifi.gguf
```

### Running Inference

```bash
# CPU inference
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100

# GPU inference (CUDA)
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100 -ngl 99

# GPU inference (Metal)
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100 -ngl 99
```

### Benchmarking

```bash
# Speed benchmark
./llama-bench -m model-q3hifi.gguf -t 4 -r 3 -p 0 -n 20

# Perplexity evaluation
./llama-perplexity -m model-q3hifi.gguf -f wikitext-2-raw/wiki.test.raw
```

## Backend Support

| Backend | Dequantization | vec_dot | Status |
|---------|----------------|---------|--------|
| CPU (AVX2) | ✅ | ✅ | Full support |
| CPU (NEON) | ✅ | ✅ | Full support |
| CUDA | ✅ | ✅ | Full support |
| Metal | ✅ | ✅ | Full support |
| SYCL | ✅ | ✅ | Full support |
| Vulkan | ✅ | ⚠️ | Dequant only (uses fallback path) |

> **Note**: For best Vulkan performance with Q3_HIFI, use CPU (`-ngl 0`) or ensure CUDA/Metal is available.

## When to Use Q3_HIFI

### ✅ Recommended For:
- Memory-constrained deployments (~25 MiB smaller than Q3_K_M)
- Edge devices with limited RAM
- Batch inference where size matters more than PPL

### ❌ Consider Alternatives If:
- Best quality at 3-bit is critical → use Q3_K_M (slightly better PPL)
- Maximum speed is critical → use Q3_K_S
- Quality is paramount → use Q4_K_M or higher

## Technical Details

### Outlier Selection Algorithm
1. Compute importance score: `score[i] = |weight[i]| × imatrix[i]`
2. Select top-8 positions by score
3. Store exact FP16 values at those positions

### Memory Layout Compatibility
The first 110 bytes of `block_q3_hifi` exactly match `block_q3_K`, enabling:
- Reuse of optimized Q3_K SIMD kernels
- Minimal code changes for backend support
- Zero-copy bulk dot product computation

### Performance Optimizations
1. **Loop unrolling**: All 8 outliers unrolled in vec_dot
2. **Pre-zeroing**: Outliers set to 0 during quantization
3. **SIMD-friendly layout**: Q3_K-compatible bit packing
4. **Factored multiplications**: d_y multiplied once per block (not per outlier)
5. **Pre-loaded FP16 values**: Better instruction pipelining

## References

- [llama.cpp Quantization Guide](../build.md)
- [Q3_K Implementation](../../ggml/src/ggml-quants.c)
- [Original GPTQ Paper](https://arxiv.org/abs/2210.17323)

