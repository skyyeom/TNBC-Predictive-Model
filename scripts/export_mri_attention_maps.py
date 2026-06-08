#!/usr/bin/env python
"""Export MRI attention maps after training a real deep MRI encoder.

This script is a safe scaffold. It does not fabricate heatmaps. Add a trained PyTorch
checkpoint and image loader before using it for publication figures.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mri_attention_maps import save_attention_overlay_placeholder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs/real/attention_maps")
    args = p.parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    save_attention_overlay_placeholder(outdir / "README_attention_maps.txt")
    print(f"Created attention-map instructions at {outdir}")

if __name__ == "__main__":
    main()
