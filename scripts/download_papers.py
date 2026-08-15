#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "references" / "papers"
DEST = ROOT
PAPERS = {
    "trackstar-2410.17413v3.pdf": {
        "url": "https://arxiv.org/pdf/2410.17413v3",
        "sha256": "fccf4e0d9a3ec9475bc0e73c5e12de6c468615f8887e605e39455f4a12accb8c",
    },
    "trak-2303.14186v2.pdf": {
        "url": "https://arxiv.org/pdf/2303.14186v2",
        "sha256": "407fc348aa2e2b22a2a5f279924d78413e04c0a8e7b61c891b013aa9cafc6993",
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
