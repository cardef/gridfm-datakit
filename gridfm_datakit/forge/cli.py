"""Command-line interface for GridFM Forge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ForgeConfig, build_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridfm-forge",
        description="Compile GridFM DataKit outputs into graph-native ML datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build graph shards, splits and provenance")
    build.add_argument("input_dir", help="Raw GridFM DataKit output directory")
    build.add_argument("output_dir", help="Destination directory")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--train-fraction", type=float, default=0.8)
    build.add_argument("--validation-fraction", type=float, default=0.1)
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument("--active-tolerance", type=float, default=1e-3)

    inspect = subparsers.add_parser("inspect", help="Print a Forge manifest")
    inspect.add_argument("dataset_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        manifest = build_dataset(
            ForgeConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                batch_size=args.batch_size,
                active_tolerance=args.active_tolerance,
            )
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        manifest_path = Path(args.dataset_dir) / "manifest.json"
        print(json.dumps(json.loads(manifest_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
