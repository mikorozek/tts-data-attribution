#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "references" / "papers"
DEST = ROOT
PAPERS = {
    "qwen3-tts-2601.15621v1.pdf": {
        "url": "https://arxiv.org/pdf/2601.15621v1",
        "sha256": "4a857188ba3478a410faacc0045cf00b40a00ecb0499617b4e9d70e7cf4b87c8",
    },
    "dailytalk-2207.01063v3.pdf": {
        "url": "https://arxiv.org/pdf/2207.01063v3",
        "sha256": "e862d63eea2ed631c3adadc5cd81acfa08a3430baa985e9b3f84636ffc61154d",
    },
    "lora-2106.09685v2.pdf": {
        "url": "https://arxiv.org/pdf/2106.09685v2",
        "sha256": "e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for filename, spec in PAPERS.items():
        target = DEST / filename
        if target.exists() and sha256(target) == spec["sha256"]:
            print(f"ok: {filename}")
            continue
        request = Request(
            spec["url"], headers={"User-Agent": "tts-data-attribution-research/0.1"}
        )
        with urlopen(request, timeout=120) as response:
            content = response.read()
        if not content.startswith(b"%PDF"):
            raise RuntimeError(f"Not a PDF: {spec['url']}")
        target.write_bytes(content)
        actual = sha256(target)
        if actual != spec["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {filename}: {actual}")
        print(f"downloaded: {filename}")


if __name__ == "__main__":
    main()
