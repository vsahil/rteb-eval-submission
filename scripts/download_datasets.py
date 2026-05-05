"""
Download all public RTEB datasets to a local cache directory.

Usage:
    python scripts/download_datasets.py [--cache-dir /path/to/cache]
"""

import argparse
from rteb.datasets import DATASET_REGISTRY
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download all RTEB public datasets")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: HuggingFace default)",
    )
    args = parser.parse_args()

    print(f"Downloading {len(DATASET_REGISTRY)} datasets...")

    for name, config in sorted(DATASET_REGISTRY.items()):
        hf_path = config["hf_path"]
        hf_subset = config.get("hf_subset", None)
        print(f"\n  Downloading: {name} ({hf_path})")

        try:
            kwargs = {"path": hf_path}
            if hf_subset:
                kwargs["name"] = hf_subset
            if args.cache_dir:
                kwargs["cache_dir"] = args.cache_dir

            load_dataset(**kwargs)
            print(f"    Done.")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
