"""Resume verified HTTP range downloads when the Hub's bulk transfer stalls.

Uses the configured Hugging Face login. Never prints signed CDN URLs or tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import time

from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    info = get_hf_file_metadata(hf_hub_url(args.repo, args.filename, revision=args.revision), token=True, timeout=30)
    if not re.fullmatch(r"[0-9a-f]{64}", info.etag or ""):
        raise RuntimeError("Expected a SHA-256 LFS checksum; no download started.")
    if Path(args.filename).name != args.filename:
        raise ValueError("Supply a filename without directories.")
    root = Path(constants.HF_HUB_CACHE) / ("models--" + args.repo.replace("/", "--")) / "snapshots" / info.commit_hash
    root.mkdir(parents=True, exist_ok=True)
    destination = root / args.filename
    if destination.exists():
        with destination.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == info.etag:
                print("Verified existing weights.", flush=True)
                return
        raise RuntimeError("Existing weights have an unexpected checksum.")
    parts = root / (args.filename + ".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    count = (info.size + chunk_size - 1) // chunk_size

    def fetch(index):
        start = index * chunk_size
        end = min(start + chunk_size, info.size) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(3):
            try:
                with requests.get(info.location, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 60)) as response:
                    expected = f"bytes {start}-{end}/{info.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("CDN did not honor the requested byte range.")
                    with path.open("wb") as handle:
                        for block in response.iter_content(256 * 1024):
                            handle.write(block)
                if path.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete byte range.")
                return path
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Chunk {index} failed: {type(exc).__name__}") from None
                time.sleep(attempt + 1)

    print(f"Downloading {info.size / 1e6:.0f} MB in {count} verified ranges.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == count:
                print(f"Completed {completed}/{count} ranges", flush=True)
    partial = root / (args.filename + ".partial")
    checksum = hashlib.sha256()
    with partial.open("wb") as output:
        for index in range(count):
            with (parts / str(index)).open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    checksum.update(block)
                    output.write(block)
    if checksum.hexdigest() != info.etag:
        raise RuntimeError("Full weight checksum mismatch; partial file retained for inspection.")
    partial.replace(destination)
    for index in range(count):
        (parts / str(index)).unlink()
    parts.rmdir()
    print(f"Verified SHA-256; weights saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
