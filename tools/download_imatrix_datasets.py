#!/usr/bin/env python3
"""Download datasets for imatrix generation."""

from typing import Any, cast

from datasets import load_dataset


def download_mathqa(output_file="mathqa-2.5k.txt", num_samples=2500) -> tuple[str, int]:
    """Download MathQA problems. Returns (filename, expected_count)."""
    print(f"Downloading MathQA dataset ({num_samples} samples)...")
    ds = load_dataset('allenai/math_qa', revision='refs/convert/parquet', split='train')
    with open(output_file, 'w') as f:
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            f.write(item['Problem'].strip() + '\n')
    print(f"  Saved to {output_file}")
    return output_file, num_samples


def download_codeparrot(output_file="codeparrot-2.5k.txt", num_samples=2500) -> tuple[str, int]:
    """Download CodeParrot code snippets. Returns (filename, expected_count)."""
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
    return output_file, num_samples


def download_wikitext(output_file="wikitext-5k.txt", num_lines=5000) -> tuple[str, int]:
    """Download WikiText samples. Returns (filename, actual_count)."""
    print(f"Downloading WikiText dataset ({num_lines} lines)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    count = 0
    with open(output_file, 'w') as f:
        for i, line in enumerate(ds['text']):
            if i >= num_lines:
                break
            if line.strip():
                f.write(line.strip() + '\n')
                count += 1
    print(f"  Saved to {output_file}")
    return output_file, count


def verify_file(filename: str, expected: int) -> bool:
    """Verify that a file has the expected number of lines."""
    with open(filename, 'r') as f:
        actual = sum(1 for _ in f)
    if actual == expected:
        print(f"  ✓ {filename}: {actual} lines")
        return True
    else:
        print(f"  ✗ {filename}: expected {expected}, got {actual}")
        return False


if __name__ == "__main__":
    results = [
        download_mathqa(),
        download_codeparrot(),
        download_wikitext(),
    ]

    print("\nVerifying downloads...")
    all_ok = all(verify_file(f, n) for f, n in results)

    if all_ok:
        print("\nDone! All files verified.")
    else:
        print("\nWarning: Some files have unexpected line counts.")
        exit(1)
