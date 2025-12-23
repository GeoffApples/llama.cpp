# Outlier Analysis Tool

Analyzes weight outlier distributions from FP32/FP16 models to inform LUT design for Q3_HIFI quantization.

## Purpose

This tool supports the Q3_HIFI development roadmap by:

1. **Phase 0**: Validating that FP32 outliers provide measurable benefit over FP16
2. **Phase 1**: Generating optimal LUT entries for sign-aware magnitude compression

## Usage

```bash
./llama-outlier-analysis model.gguf [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--imatrix <file>` | Load importance matrix for weighted outlier selection |
| `--output <file>` | Output file for outlier data (default: `outliers.txt`) |
| `--n-outliers <n>` | Number of outliers per block (default: 6) |
| `--n-lut <n>` | Number of LUT entries to generate (default: 128) |
| `--verbose` | Print detailed per-tensor statistics |

## Example

```bash
# Basic analysis
./llama-outlier-analysis Qwen3-0.6B-F16.gguf

# With importance matrix
./llama-outlier-analysis Qwen3-0.6B-F16.gguf --imatrix qwen3-0.6b.imatrix --output outliers.txt

# Generate 128-entry LUT
./llama-outlier-analysis Qwen3-0.6B-F16.gguf --n-lut 128
```

## Output

The tool generates:

1. **Statistics**: Min, max, mean, median, std of outlier magnitudes
2. **Percentiles**: 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%
3. **LUT entries**: Optimal lookup table values (k-means clustered)
4. **Per-layer breakdown**: Statistics by layer type (attn_v, ffn_down, etc.)

## Phase 0 Workflow

1. **Generate imatrix** (recommended):
   ```bash
   ./llama-imatrix -m model.gguf -f calibration.txt -o model.imatrix --chunks 5000
   ```

2. **Quantize with Q3_HIFI_F32_RAW**:
   ```bash
   ./quantize model.gguf model-q3hifi-f32.gguf Q3_HIFI_F32_RAW --imatrix model.imatrix
   ```

3. **Benchmark perplexity**:
   ```bash
   ./perplexity -m model-q3hifi-f32.gguf -f wikitext-2-raw/wiki.test.raw
   ```

4. **Compare against Q3_HIFI (FP16)**:
   - If math PPL improves ≥ 0.3, proceed to Phase 1
   - If no improvement, FP32 outliers may not be worth the size increase

## Expected Results

Based on typical Qwen3 models:
- 99% of outliers fall in [0.006, 0.16] (normalized)
- Mean absolute outlier: ~0.04
- A 128-entry LUT should capture >99% of the distribution

## See Also

- [F32_ROADMAP.md](../../F32_ROADMAP.md) - Full development roadmap
- [Q3_HIFI.md](../../docs/quantization/Q3_HIFI.md) - Q3_HIFI documentation

