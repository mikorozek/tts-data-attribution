from __future__ import annotations

import argparse
from pathlib import Path

from tts_data_attribution.dataset import DailyTalkDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dailytalk",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dailytalk.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataset = DailyTalkDataset.from_directory(arguments.data_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_jsonl(arguments.output)
    print(f"{len(dataset)} examples written to {arguments.output}")


if __name__ == "__main__":
    main()
