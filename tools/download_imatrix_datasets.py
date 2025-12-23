#!/usr/bin/env python3
"""Download datasets for imatrix generation."""

from typing import Any, cast

from datasets import load_dataset


def download_mathqa(output_file="mathqa-2.5k.txt", num_samples=2500):
    """Download MathQA problems."""
    print(f"Downloading MathQA dataset ({num_samples} samples)...")
    ds = load_dataset('allenai/math_qa', revision='refs/convert/parquet', split='train')
    with open(output_file, 'w') as f:
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            f.write(item['Problem'].strip() + '\n')
    print(f"  Saved to {output_file}")


def download_codeparrot(output_file="codeparrot-2.5k.txt", num_samples=2500):
    """Download CodeParrot code snippets."""
    print(f"Downloading CodeParrot dataset ({num_samples} samples)...")
    ds = load_dataset('codeparrot/codeparrot-valid-v2-near-dedup', split='train', streaming=True)
    with open(output_file, 'w') as f:
        count = 0
        for item in ds:
            if count >= num_samples:
                break
            code = cast(dict[str, Any], item)['content'].strip()
            if code and len(code) > 20:  # skip tiny snippets
                f.write(code + '\n')
                count += 1
    print(f"  Saved to {output_file}")


def download_wikitext(output_file="wikitext-5k.txt", num_lines=5000):
    """Download WikiText samples."""
    print(f"Downloading WikiText dataset ({num_lines} lines)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    with open(output_file, 'w') as f:
        for i, line in enumerate(ds['text']):
            if i >= num_lines:
                break
            if line.strip():
                f.write(line.strip() + '\n')
    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    download_mathqa()
    download_codeparrot()
    download_wikitext()
    print("Done!")
