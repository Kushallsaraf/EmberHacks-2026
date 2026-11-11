"""Fetch only the bodyweight-squat videos from the Mendeley fitness dataset.

Source: https://data.mendeley.com/datasets/kgbb3yn47p/3 (CC BY 4.0). The
"Exercise-wise" archive is ~3.1 GB; HTTP range requests read its zip index and
the selected members only, so the full archive is never downloaded.
"""
from __future__ import annotations

import argparse
import io
import logging
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

LOG = logging.getLogger(__name__)
ARCHIVE_URL = ("https://data.mendeley.com/public-files/datasets/kgbb3yn47p/files/"
               "a31fa913-6f29-4c31-b1a6-414a718eebf4/file_downloaded")
ARCHIVE_SIZE = 3113452092
# python.org builds ship without root certificates.
CONTEXT = ssl.create_default_context(cafile=certifi.where())
# Mendeley rejects urllib's default User-Agent.
HEADERS = {"User-Agent": "spotter-ml-dataset-fetch/1.0"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        # Resolve the redirect once; the storage URL serves the bytes.
        with urllib.request.urlopen(urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"}), timeout=60, context=CONTEXT) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        request = urllib.request.Request(self.url, headers={**HEADERS, "Range": f"bytes={self.position}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=120, context=CONTEXT) as response:
            data = response.read()
        self.position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def squat_members(archive: zipfile.ZipFile, keyword: str) -> list[zipfile.ZipInfo]:
    return [m for m in archive.infolist() if not m.is_dir() and keyword in m.filename.lower()
            and m.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ml/data/downloads/mendeley_kgbb3yn47p_v3"))
    parser.add_argument("--keyword", default="squat", help="Case-insensitive path filter.")
    parser.add_argument("--list", action="store_true", help="List archive contents and exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    remote = HttpRangeFile(ARCHIVE_URL)
    if remote.size != ARCHIVE_SIZE:
        parser.exit(2, f"Unexpected archive size {remote.size}; the dataset may have changed.\n")
    # A large buffer keeps member reads to a few requests each.
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=8 * 1024 * 1024)) as archive:
        if args.list:
            for m in archive.infolist():
                print(f"{m.file_size:>12} {m.filename}")
            return
        members = squat_members(archive, args.keyword)
        LOG.info("%d matching videos, %.2f GB", len(members), sum(m.compress_size for m in members) / 1e9)
        for m in members:
            target = args.output / m.filename
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            # zipfile verifies each member's CRC-32 while reading.
            with archive.open(m) as source, partial.open("wb") as sink:
                while block := source.read(4 * 1024 * 1024):
                    sink.write(block)
            partial.rename(target)
            LOG.info("Saved %s (%.1f MB)", target, m.file_size / 1e6)


if __name__ == "__main__":
    main()
