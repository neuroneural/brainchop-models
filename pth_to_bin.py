#!/usr/bin/env python3
"""Convert PyTorch .pth model files to raw .bin weight files.

Extracts all weight tensors from a PyTorch state dict and writes them
as a single concatenated float32 binary file.

Usage:
    python pth_to_bin.py model.pth                  # writes model.bin next to model.pth
    python pth_to_bin.py model.pth -o weights.bin   # writes to specific output path
    python pth_to_bin.py --all                       # convert all .pth missing a .bin
    python pth_to_bin.py --all --force               # reconvert all .pth even if .bin exists
"""
import argparse
import struct
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


def convert_pth_to_bin(pth_path: Path, bin_path: Path, verbose: bool = True) -> None:
    """Convert a .pth state dict to a raw float32 .bin file."""
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    total_params = 0
    with open(bin_path, "wb") as f:
        for key, tensor in state_dict.items():
            arr = tensor.detach().float().numpy()
            f.write(arr.tobytes())
            total_params += arr.size
            if verbose:
                print(f"  {key}: {list(arr.shape)}")

    size_kb = bin_path.stat().st_size / 1024
    if verbose:
        print(f"  Total parameters: {total_params:,}")
        print(f"  Output: {bin_path} ({size_kb:.1f} KB)")


def find_pth_files(root: Path) -> list[Path]:
    """Find all .pth files under root directory."""
    return sorted(root.rglob("*.pth"))


def main():
    parser = argparse.ArgumentParser(description="Convert .pth to .bin weight files")
    parser.add_argument("input", nargs="?", type=Path, help="Input .pth file path")
    parser.add_argument("-o", "--output", type=Path, help="Output .bin file path")
    parser.add_argument("--all", action="store_true", help="Convert all .pth files missing .bin")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .bin files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-tensor output")
    args = parser.parse_args()

    if args.all:
        root = Path(__file__).parent
        pth_files = find_pth_files(root)
        if not pth_files:
            print("No .pth files found.")
            return

        for pth_path in pth_files:
            bin_path = pth_path.with_suffix(".bin")
            if bin_path.exists() and not args.force:
                print(f"Skipping {pth_path} (.bin already exists, use --force to overwrite)")
                continue
            print(f"Converting {pth_path}...")
            convert_pth_to_bin(pth_path, bin_path, verbose=not args.quiet)
            print()
    elif args.input:
        if not args.input.exists():
            print(f"Error: {args.input} not found", file=sys.stderr)
            sys.exit(1)
        bin_path = args.output or args.input.with_suffix(".bin")
        print(f"Converting {args.input}...")
        convert_pth_to_bin(args.input, bin_path, verbose=not args.quiet)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
