from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILYTALK_FILE_ID = "1nPrfJn3TcIVPc0Uf5tiAXUYLJceb_5k-"
DAILYTALK_ARCHIVE_BYTES = 5_341_371_062


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data" / "downloads" / "dailytalk.zip",
    )
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != DAILYTALK_ARCHIVE_BYTES:
        gdown.download(
            id=DAILYTALK_FILE_ID,
            output=str(path),
            resume=True,
        )
    if path.stat().st_size != DAILYTALK_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Expected {DAILYTALK_ARCHIVE_BYTES} bytes, found {path.stat().st_size}"
        )


def main() -> None:
    arguments = parse_arguments()
    dataset_path = arguments.data_root / "dailytalk"
    if dataset_path.exists():
        print(dataset_path)
        return

    download_archive(arguments.archive)
    arguments.data_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(arguments.archive) as archive:
        archive.extractall(arguments.data_root)

    if not dataset_path.is_dir():
        raise RuntimeError(f"Archive did not create {dataset_path}")
    if not arguments.keep_archive:
        arguments.archive.unlink()
    print(dataset_path)


if __name__ == "__main__":
    main()
