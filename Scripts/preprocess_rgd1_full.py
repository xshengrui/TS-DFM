import argparse
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from Data.rgd1_preprocess import preprocess_rgd1_zip


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert the full RGD1 ZIP into validated PyTorch shards."
    )
    parser.add_argument("--source-zip", required=True, help="Path to RGD1.zip")
    parser.add_argument(
        "--output-dir",
        default="Data/processed/rgd1_full",
        help="New cache directory to create",
    )
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--expected-count", type=int, default=176898)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = preprocess_rgd1_zip(
        args.source_zip,
        args.output_dir,
        shard_size=args.shard_size,
        expected_count=args.expected_count,
    )
    print(
        f"Processed {manifest['record_count']} RGD1 reactions into "
        f"{len(manifest['shards'])} shards at {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
