#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix (const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

// ====================== Q4_HIFI Adaptive Quantization System ======================
// Three-strategy approach to make Q4_HIFI competitive across all model sizes:
//   Strategy 1: Hybrid Base Precision - Use Q5_K/Q6_K base for 4B+ models
//   Strategy 2: Architecture-Aware Selection - Adjust based on layer count, not just params
//   Strategy 3: Dynamic Outlier Budget - Focus outliers on early/middle layers
//
// Analysis shows Q4_HIFI excels at ≤3B but underperforms at 4B+ because:
//   - Q4_K_M uses Q6_K on sensitive layers → higher base precision where it matters
//   - Deeper models (36+ layers) accumulate more quantization error
//   - Outlier preservation alone isn't enough at scale
//
// Solution: Match Q4_K_M's mixed-precision approach while preserving outlier benefits

// ====================== STRATEGY 1: Selective Hybrid Base Precision ======================
// For models >3B, use higher base precision (Q5_K/Q6_K) ONLY for critical tensors
// Non-critical tensors use Q4_K to maintain speed while preserving quality where it matters
//
// Critical tensors (get Q5_K/Q6_K base):
//   - attn_v, ffn_down: Already have Q4_HIFI outliers, base doesn't affect these
//   - lm_head/output: Critical for vocabulary prediction
//   - token_embd: First layer, impacts all downstream activations
//   - attn_output: Projection layer, high sensitivity
//
// Non-critical tensors (use Q4_K base for speed):
//   - ffn_gate, ffn_up: Less sensitive, Q4_K is sufficient
//   - attn_q, attn_k: Query/key projections, less impact than value

// Check if a tensor is critical and needs higher base precision
static bool q4_hifi_is_critical_tensor(const std::string & name) {
    // Output/vocabulary tensors - critical for prediction quality
    if (name.find("output.weight") != std::string::npos) return true;
    if (name.find("lm_head") != std::string::npos) return true;
    
    // Token embeddings - first layer, high impact
    if (name.find("token_embd") != std::string::npos) return true;
    if (name.find("tok_embeddings") != std::string::npos) return true;
    if (name.find("embed_tokens") != std::string::npos) return true;
    
    // Attention output projection - sensitive layer
    if (name.find("attn_output") != std::string::npos) return true;
    
    // Note: attn_v and ffn_down already get Q4_HIFI treatment (outliers)
    // so their base precision is handled by the Q4_HIFI quantization itself
    
    return false;
}

// Get base quantization type for a specific tensor (selective approach)
static ggml_type q4_hifi_get_tensor_base_quant(const std::string & name, int64_t param_count) {
    // For small models (≤3B), use Q4_K everywhere - works great
    if (param_count <= 3000000000LL) {
        return GGML_TYPE_Q4_K;
    }
    
    // For larger models, only critical tensors get higher precision
    if (q4_hifi_is_critical_tensor(name)) {
        if (param_count <= 10000000000LL) {
            return GGML_TYPE_Q5_K;   // 3B-10B: Q5_K for critical tensors
        } else {
            return GGML_TYPE_Q6_K;   // >10B: Q6_K for critical tensors
        }
    }
    
    // Non-critical tensors: use Q4_K for speed (same as Q4_K_M behavior)
    return GGML_TYPE_Q4_K;
}

// Legacy function for global base type (used by quantize_state_impl)
static ggml_type q4_hifi_get_base_quant_type(int64_t param_count) {
    if (param_count <= 3000000000LL) {
        return GGML_TYPE_Q4_K;   // ≤3B: Pure 4-bit
    } else if (param_count <= 10000000000LL) {
        return GGML_TYPE_Q5_K;   // 3B-10B: Q5_K (for critical tensors)
    } else {
        return GGML_TYPE_Q6_K;   // >10B: Q6_K (for critical tensors)
    }
}

// ====================== STRATEGY 2: Architecture-Aware Outlier Selection ======================
// Adjust FFN coverage based on layer count (depth), not just parameter count
// Deep models (40+ layers) accumulate more quantization error → need different strategy

static float q4_hifi_get_ffn_coverage_by_architecture(int64_t param_count, int n_layers) {
    // Architecture-aware: layer depth matters more than raw param count
    // Qwen-4B has 36 layers vs Qwen-1.7B's 28 → 29% deeper = more error accumulation
    
    // For very large models (>70B), skip FFN regardless of layer count
    if (param_count > 70000000000LL) {
        return 0.0f;
    }
    
    if (n_layers <= 30) {
        // Shallow models (≤30 layers): Full 50% FFN coverage - outliers very effective
        return 0.50f;
    } else if (n_layers <= 40) {
        // Medium models (31-40 layers): Reduce to 30% FFN coverage
        // Focus outlier budget on attn_v which has higher impact per layer
        return 0.30f;
    } else if (n_layers <= 60) {
        // Deep models (41-60 layers): Only 15% FFN coverage
        return 0.15f;
    } else {
        // Very deep models (60+ layers): Skip FFN, focus entirely on attn_v
        return 0.0f;
    }
}

// Legacy function for backward compatibility (still considers param count)
static float q4_hifi_get_ffn_coverage(int64_t param_count) {
    if (param_count <= 10000000000LL)   return 0.50f;  // ≤10B:  50% of ffn_down layers
    if (param_count <= 70000000000LL)   return 0.25f;  // 10B-70B: 25% of ffn_down layers
    return 0.0f;                                        // >70B: 0% (attn_v only)
}

// ====================== STRATEGY 3: Dynamic Outlier Budget Allocation ======================
// Allocate more outliers to early/middle layers, fewer to final layers
// Research shows early layers dominate attention quality; late layers are less sensitive

static int q4_hifi_get_layer_outliers(int base_outliers, int layer_idx, int total_layers) {
    // Calculate depth ratio (0.0 = first layer, 1.0 = last layer)
    float depth_ratio = (float)layer_idx / (float)total_layers;
    
    if (depth_ratio < 0.4f) {
        // Early layers (first 40%): Full outlier budget - most sensitive to quantization
        return base_outliers;
    } else if (depth_ratio < 0.7f) {
        // Middle layers (40-70%): 75% of budget
        return (int)(base_outliers * 0.75f);
    } else {
        // Late layers (final 30%): 50% of budget - less impact on output quality
        return base_outliers / 2;
    }
}

// Base outlier count based on model size (unchanged from original)
static int q4_hifi_get_base_outliers(int64_t param_count) {
    if (param_count <= 3000000000LL)  return 8;   // ≤3B:    8 outliers
    if (param_count <= 30000000000LL) return 10;  // 3B-30B: 10 outliers
    if (param_count <= 70000000000LL) return 12;  // 30B-70B: 12 outliers
    return 16;                                     // >70B:   16 outliers
}

static int q4_hifi_get_massive_outliers(int64_t param_count) {
    return q4_hifi_get_base_outliers(param_count) * 2;
}

// Check if a tensor is a "massive" layer that benefits from extra outliers
static bool q4_hifi_is_massive_tensor(const std::string & name) {
    // Output/lm_head tensors - critical for vocabulary prediction
    if (name.find("output.weight") != std::string::npos) return true;
    if (name.find("lm_head") != std::string::npos) return true;
    
    // Token embeddings - first layer, high impact on all downstream
    if (name.find("token_embd.weight") != std::string::npos) return true;
    if (name.find("tok_embeddings") != std::string::npos) return true;
    
    return false;
}

// Extract layer index from tensor name (e.g., "blk.15.attn_v.weight" → 15)
static int q4_hifi_get_layer_from_name(const std::string & name) {
    // Look for "blk.N." pattern
    size_t blk_pos = name.find("blk.");
    if (blk_pos == std::string::npos) {
        return -1;  // Not a layer tensor
    }
    
    size_t num_start = blk_pos + 4;
    size_t num_end = name.find('.', num_start);
    if (num_end == std::string::npos) {
        return -1;
    }
    
    std::string num_str = name.substr(num_start, num_end - num_start);
    return std::stoi(num_str);
}

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_k_quantized = 0;
    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    // ====================== Q4_HIFI Adaptive Configuration ======================
    // Three-strategy system for competitive performance across all model sizes
    int64_t total_params = 0;
    int q4_hifi_n_layers = 0;              // Strategy 2: Layer count for architecture-aware selection
    int q4_hifi_base_outliers = 16;
    int q4_hifi_massive_outliers = 32;
    float q4_hifi_ffn_coverage = 0.5f;     // Fraction of ffn_down layers to use Q4_HIFI
    ggml_type q4_hifi_base_type = GGML_TYPE_Q4_K;  // Strategy 1: Adaptive base precision

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
    
    // Initialize Q4_HIFI with all three strategies based on model architecture
    void init_q4_hifi_adaptive(int64_t param_count, int n_layers) {
        total_params = param_count;
        q4_hifi_n_layers = n_layers;
        
        // Strategy 1: Determine base quantization type for critical tensors
        // (non-critical tensors will use Q4_K regardless)
        q4_hifi_base_type = q4_hifi_get_base_quant_type(param_count);
        
        // Strategy 2: Use architecture-aware FFN coverage (considers layer depth)
        q4_hifi_ffn_coverage = q4_hifi_get_ffn_coverage_by_architecture(param_count, n_layers);
        
        // Base outlier counts (will be adjusted per-layer by Strategy 3)
        q4_hifi_base_outliers = q4_hifi_get_base_outliers(param_count);
        q4_hifi_massive_outliers = q4_hifi_get_massive_outliers(param_count);
        
        // Log the adaptive configuration
        const char * base_type_name = ggml_type_name(q4_hifi_base_type);
        LLAMA_LOG_INFO("\n%s: Q4_HIFI Adaptive System Initialized\n", __func__);
        LLAMA_LOG_INFO("%s:   Model: %.2fB params, %d layers\n", __func__, param_count / 1e9, n_layers);
        LLAMA_LOG_INFO("%s:   Strategy 1 - Selective Base Precision:\n", __func__);
        LLAMA_LOG_INFO("%s:       Critical tensors (output, embed, attn_output): %s\n", __func__, base_type_name);
        LLAMA_LOG_INFO("%s:       Non-critical tensors (ffn_gate/up, attn_q/k): Q4_K (for speed)\n", __func__);
        LLAMA_LOG_INFO("%s:   Strategy 2 - FFN Coverage: %.0f%% (architecture-aware)\n", __func__, q4_hifi_ffn_coverage * 100);
        LLAMA_LOG_INFO("%s:   Strategy 3 - Outlier Budget: %d base, %d massive (layer-adaptive)\n", __func__, q4_hifi_base_outliers, q4_hifi_massive_outliers);
        
        // Log expected improvements for 4B+ models
        if (param_count > 3000000000LL) {
            LLAMA_LOG_INFO("%s:   Note: Selective Q5_K reduces size/improves speed vs uniform Q5_K\n", __func__);
        }
    }
    
    // Legacy initializer for backward compatibility
    void init_q4_hifi_outliers(int64_t param_count) {
        init_q4_hifi_adaptive(param_count, (int)model.hparams.n_layer);
    }
    
    // Get outlier count for a specific tensor (Strategy 3: layer-adaptive)
    int get_q4_hifi_outliers(const std::string & tensor_name) const {
        // Massive tensors always get full budget (they're not layer-specific)
        if (q4_hifi_is_massive_tensor(tensor_name)) {
            return q4_hifi_massive_outliers;
        }
        
        // Strategy 3: Adjust outliers based on layer depth
        int layer_idx = q4_hifi_get_layer_from_name(tensor_name);
        if (layer_idx >= 0 && q4_hifi_n_layers > 0) {
            return q4_hifi_get_layer_outliers(q4_hifi_base_outliers, layer_idx, q4_hifi_n_layers);
        }
        
        return q4_hifi_base_outliers;
    }
    
    // Get base quantization type for a specific tensor (Strategy 1: selective)
    ggml_type get_q4_hifi_tensor_base_type(const std::string & tensor_name) const {
        return q4_hifi_get_tensor_base_quant(tensor_name, total_params);
    }
    
    // Get the base quantization type for non-HIFI tensors (Strategy 1)
    ggml_type get_q4_hifi_base_type() const {
        return q4_hifi_base_type;
    }
};

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs.has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
                // Q4_HIFI: Use Q6_K for output tensor (critical for vocabulary prediction quality)
                new_type = GGML_TYPE_Q6_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && name.find("attn_k.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_HIFI) {
            // Adaptive Q3_HIFI: use Q3_HIFI for ALL attn_v layers (consistently sensitive)
            new_type = GGML_TYPE_Q3_HIFI;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // Strategic Q4_HIFI: FP16 outliers ONLY for early layers (0-10)
            // Early layers dominate attention quality; late layers use Q4_K for speed
            int layer_idx = q4_hifi_get_layer_from_name(name);
            if (layer_idx >= 0 && layer_idx <= 10) {
                new_type = GGML_TYPE_Q4_HIFI;  // FP16 outliers for early layers
            } else {
                new_type = GGML_TYPE_Q4_K;  // Q4_K for late layers (speed)
            }
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // Revolutionary Q4_HIFI_RESIDUAL: Apply only to early layers (0-10)
            // Early layers dominate attention quality; late layers are less sensitive
            int layer_idx = q4_hifi_get_layer_from_name(name);
            if (layer_idx >= 0 && layer_idx <= 10) {
                new_type = GGML_TYPE_Q4_HIFI_RESIDUAL;  // INT8 residuals for early layers
            } else {
                new_type = GGML_TYPE_Q4_K;  // Q4_K for late layers (speed)
            }
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k.weight") != std::string::npos) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // attn_k is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // attn_k is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
    } else if (name.find("attn_q.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // attn_q is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // attn_q is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_HIFI) {
            // Adaptive Q3_HIFI: use Q3_HIFI for first 1/3 of ffn_down layers (most sensitive)
            new_type = i_layer < n_layer/3 ? GGML_TYPE_Q3_HIFI
                     : use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // Strategic Q4_HIFI: FP16 outliers ONLY for early layers (0-10)
            // Early ffn_down layers have highest impact on MLP quality
            if (i_layer <= 10) {
                new_type = GGML_TYPE_Q4_HIFI;  // FP16 outliers for early layers
            } else {
                new_type = GGML_TYPE_Q4_K;  // Q4_K for late layers (speed)
            }
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // Revolutionary Q4_HIFI_RESIDUAL: Apply only to early layers (0-10)
            // Early ffn_down layers have highest impact on MLP quality
            if (i_layer <= 10) {
                new_type = GGML_TYPE_Q4_HIFI_RESIDUAL;  // INT8 residuals for early layers
            } else {
                new_type = GGML_TYPE_Q4_K;  // Q4_K for late layers (speed)
            }
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_HIFI) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
                    // Strategic Q4_HIFI: FP16 outliers for all attn_output (always critical)
                    new_type = GGML_TYPE_Q4_HIFI;
                }
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
                    // attn_output is critical → use Q4_HIFI_RESIDUAL for quality
                    new_type = GGML_TYPE_Q4_HIFI_RESIDUAL;
                }
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M ||
            ftype == LLAMA_FTYPE_MOSTLY_Q3_HIFI) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // attn_qkv is mixed (contains v) → use Q4_K for speed
            // Note: Individual attn_v layers handled above
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // attn_qkv contains V which is sensitive → use Q4_K (speed)
            // Note: Individual attn_v layers get Q4_HIFI_RESIDUAL above
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // ffn_gate is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // ffn_gate is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        ++qs.i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
            // ffn_up is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL) {
            // ffn_up is non-critical → use Q4_K for speed
            new_type = GGML_TYPE_Q4_K;
        }
        ++qs.i_ffn_up;
    }

    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // IK: let's remove this, else Q2_K is almost the same as Q3_K_S
    //else if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // This can be used to reduce the size of the Q5_K_S model.
    // The associated PPL increase is fully in line with the size reduction
    //else {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) new_type = GGML_TYPE_Q4_K;
    //}
    bool convert_incompatible_tensor = false;
    {
        const int64_t nx = tensor->ne[0];
        const int64_t ny = tensor->ne[1];
        const int64_t qk_k = ggml_blck_size(new_type);

        if (nx % qk_k != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %" PRId64 " x %" PRId64 " are not divisible by %" PRId64 ", required for %s", __func__, nx, ny, qk_k, ggml_type_name(new_type));
            convert_incompatible_tensor = true;
        } else {
            ++qs.n_k_quantized;
        }
    }

    if (convert_incompatible_tensor) {
        switch (new_type) {
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:  new_type = GGML_TYPE_Q4_0; break;  // TODO: use a symmetric type instead
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
        }
        if (tensor->ne[0] % ggml_blck_size(new_type) != 0) {
            new_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
        ++qs.n_fallback;
    }

    return new_type;
}

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: default_type = GGML_TYPE_MXFP4; break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   default_type = GGML_TYPE_TQ1_0;   break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   default_type = GGML_TYPE_TQ2_0;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_HIFI: default_type = GGML_TYPE_Q3_K;    break; // Adaptive: Q3_K base, Q3_HIFI on sensitive layers
        case LLAMA_FTYPE_MOSTLY_Q4_HIFI: default_type = GGML_TYPE_Q4_K;    break; // Adaptive: Q4_K base, Q4_HIFI on sensitive layers
        case LLAMA_FTYPE_MOSTLY_Q4_HIFI_RESIDUAL: default_type = GGML_TYPE_Q4_K; break; // Revolutionary: Q4_K base + INT8 residuals on early layers

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto * v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    quantize_state_impl qs(model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        prune_list = *static_cast<const std::vector<int> *>(params->prune_layers);
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.meta.get());
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)std::abs(o.val_i64));
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // Count total parameters and gather tensor info
    int64_t total_param_count = 0;
    for (const auto * it : tensors) {
        const struct ggml_tensor * tensor = it->tensor;

        const std::string name = ggml_get_name(tensor);
        
        // Count parameters (number of elements in weight tensors)
        if (name.find(".weight") != std::string::npos) {
            total_param_count += ggml_nelements(tensor);
        }

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }
    
    // Initialize Q4_HIFI parameter-based outlier counts
    if (ftype == LLAMA_FTYPE_MOSTLY_Q4_HIFI) {
        qs.init_q4_hifi_outliers(total_param_count);
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // populate the original tensors so we get an initial meta data
    for (const auto * it : tensors) {
        uint16_t i_split = params->keep_split ? it->idx : 0;
        ggml_tensor * tensor = it->tensor;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    const auto tn = LLM_TN(model.arch);
    new_ofstream(0);
    for (const auto * it : tensors) {
        const auto & weight = *it;
        ggml_tensor * tensor = weight.tensor;
        if (weight.idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D and 3D tensors (experts)
        quantize &= (ggml_n_dims(tensor) >= 2);

        // do not quantize norm tensors
        quantize &= name.find("_norm.weight") == std::string::npos;

        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // these are very small (e.g. 4x4)
        quantize &= name.find("altup")  == std::string::npos;
        quantize &= name.find("laurel") == std::string::npos;

        // these are not too big so keep them as it is
        quantize &= name.find("per_layer_model_proj") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba's small yet 2D weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
        quantize &= name.find("shortconv.conv.weight") == std::string::npos;

        // do not quantize RWKV's small yet 2D weights
        quantize &= name.find("time_mix_first.weight") == std::string::npos;
        quantize &= name.find("time_mix_w0.weight") == std::string::npos;
        quantize &= name.find("time_mix_w1.weight") == std::string::npos;
        quantize &= name.find("time_mix_w2.weight") == std::string::npos;
        quantize &= name.find("time_mix_v0.weight") == std::string::npos;
        quantize &= name.find("time_mix_v1.weight") == std::string::npos;
        quantize &= name.find("time_mix_v2.weight") == std::string::npos;
        quantize &= name.find("time_mix_a0.weight") == std::string::npos;
        quantize &= name.find("time_mix_a1.weight") == std::string::npos;
        quantize &= name.find("time_mix_a2.weight") == std::string::npos;
        quantize &= name.find("time_mix_g1.weight") == std::string::npos;
        quantize &= name.find("time_mix_g2.weight") == std::string::npos;
        quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
        quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
        quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

        // do not quantize relative position bias (T5)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;

        // do not quantize specific multimodal tensors
        quantize &= name.find(".position_embd.") == std::string::npos;

        ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (!params->pure && ggml_is_quantized(default_type)) {
                int fallback = qs.n_fallback;
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);
                // unless the user specifies a type, and the tensor geometry will not require fallback quantisation
                if (params->tensor_types && qs.n_fallback - fallback == 0) {
                    const std::vector<tensor_quantization> & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
                    const std::string tensor_name(tensor->name);
                    for (const auto & [tname, qtype] : tensor_types) {
                        if (std::regex pattern(tname); std::regex_search(tensor_name, pattern)) {
                            if  (qtype != new_type) {
                                LLAMA_LOG_DEBUG("(overriding %s) ", ggml_type_name(new_type));
                                new_type = qtype; // if two or more types are specified for the same tensor, the last match wins
                            }
                        }
                    }
                }
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type = params->output_tensor_type;
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor->type != new_type;
        }

        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(remap_imatrix(tensor->name, mapped));
                if (it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                        // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                        // this is a significant error and it may be good idea to abort the process if this happens,
                        // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                        // tok_embd should be ignored in this case, since it always causes this warning
                        if (name != tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                        }
                    }
                }
            }
            if ((new_type == GGML_TYPE_IQ2_XXS ||
                 new_type == GGML_TYPE_IQ2_XS  ||
                 new_type == GGML_TYPE_IQ2_S   ||
                 new_type == GGML_TYPE_IQ1_S   ||
                (new_type == GGML_TYPE_IQ1_M && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight"))  ||
                (new_type == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(tensor->name, "token_embd.weight") != 0)) && !imatrix) {
                LLAMA_LOG_ERROR("\n\n============================================================\n");
                LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                LLAMA_LOG_ERROR("============================================================\n\n");
                throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
            }

            float * f32_data;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            // Q4_HIFI: Set per-tensor outlier count before quantization
            if (new_type == GGML_TYPE_Q4_HIFI) {
                int outliers = qs.get_q4_hifi_outliers(name);
                ggml_q4_hifi_set_outlier_count(outliers);
                LLAMA_LOG_INFO("converting to %s (%d outliers) .. ", ggml_type_name(new_type), outliers);
            } else {
                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
            }
            fflush(stdout);

            if (work.size() < (size_t)nelements * 4) {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();

            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows = tensor->ne[1];

            static const int64_t min_chunk_size = 32 * 512;
            const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

            const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
            const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
            const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

            // quantize each expert separately since they have different importance matrices
            new_size = 0;
            for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

                new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);

                // TODO: temporary sanity check that the F16 -> MXFP4 is lossless
#if 0
                if (new_type == GGML_TYPE_MXFP4) {
                    auto * x = f32_data_03;

                    //LLAMA_LOG_INFO("nrows = %d, n_per_row = %d\n", nrows, n_per_row);
                    std::vector<float> deq(nrows*n_per_row);
                    const ggml_type_traits * qtype = ggml_get_type_traits(new_type);
                    qtype->to_float(new_data_03, deq.data(), deq.size());

                    double err = 0.0f;
                    for (int i = 0; i < (int) deq.size(); ++i) {
                        err += fabsf(deq[i] - x[i]);
                        //if (fabsf(deq[i] - x[i]) > 0.00001 && i < 256) {
                        if (deq[i] != x[i]) {
                            LLAMA_LOG_INFO("deq[%d] = %f, x[%d] = %f\n", i, deq[i], i, x[i]);
                        }
                    }
                    //LLAMA_LOG_INFO("err = %f\n", err);
                    GGML_ASSERT(err == 0.00000);
                }
#endif
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size);
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB\n", __func__, total_size_new/1024.0/1024.0);

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
