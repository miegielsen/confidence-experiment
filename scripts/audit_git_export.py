"""Audit staged experiment files without printing credential contents.

Run after ``git add --all`` and before committing/pushing. Reads Git blobs rather
than tens of thousands of individual Google Drive files. This is a focused
credential check, not a guarantee that arbitrary secrets can be recognized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MODEL_SUFFIXES = {
    ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".safetensors", ".onnx",
    ".bin", ".h5", ".hdf5", ".ckpt",
}
CREDENTIAL_SUFFIXES = {".clixml", ".key", ".pem", ".pfx", ".p12"}
PATTERNS = {
    "provider_api_key": re.compile(rb"\bsk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "literal_bearer_token": re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9_.~-]{30,}"),
    "literal_json_key": re.compile(
        rb'(?i)"(?:api_key|apikey|access_token|refresh_token|client_secret)"\s*:\s*"[A-Za-z0-9_./+=-]{20,}"'
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/repository-export-audit.json")
    args = parser.parse_args()
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip())
    report_path = Path(args.report).as_posix()
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    entries = []
    findings = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        path = path_bytes.decode("utf-8")
        if path == report_path:
            continue  # Keep the audit stable when its own report is staged.
        p = Path(path)
        if stage != "0" or mode not in {"100644", "100755"}:
            findings.append({"path": path, "reason": "unsupported_index_entry"})
        if p.suffix.lower() in MODEL_SUFFIXES | CREDENTIAL_SUFFIXES:
            findings.append({"path": path, "reason": "excluded_object_or_credential_file"})
        if any(part in {".venv", "venv", "secrets", "credentials", "__pycache__"} for part in p.parts):
            findings.append({"path": path, "reason": "excluded_directory"})
        if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
            findings.append({"path": path, "reason": "environment_file"})
        entries.append((oid, path))

    # Test against any credential already present in this process, without ever
    # printing it or copying it to the report.
    env_values = [v.encode() for k, v in os.environ.items()
                  if any(x in k.upper() for x in ("API_KEY", "ACCESS_TOKEN", "GITHUB_TOKEN", "HF_TOKEN"))
                  and len(v) >= 16]
    counts = Counter()
    sizes = Counter()
    total = 0
    text_count = 0
    largest = []
    child = subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert child.stdin is not None and child.stdout is not None
    try:
        for oid, path in entries:
            child.stdin.write(oid.encode() + b"\n")
            child.stdin.flush()
            header = child.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise RuntimeError(f"Expected blob for {path}")
            length = int(header[2])
            data = child.stdout.read(length)
            if len(data) != length or child.stdout.read(1) != b"\n":
                raise RuntimeError(f"Incomplete blob for {path}")
            top = path.split("/", 1)[0]
            counts[top] += 1
            sizes[top] += length
            total += length
            largest.append((length, path))
            if length >= 100 * 1024**2:
                findings.append({"path": path, "reason": "github_regular_blob_limit"})
            if b"\0" not in data[:8192]:
                text_count += 1
                for label, pattern in PATTERNS.items():
                    if pattern.search(data):
                        findings.append({"path": path, "reason": label})
                if any(value in data for value in env_values):
                    findings.append({"path": path, "reason": "credential_from_environment"})
        child.stdin.close()
        if child.wait() != 0:
            raise RuntimeError("git cat-file failed")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    report = {
        "scope": "staged files, excluding this generated audit report",
        "staged_file_count": len(entries),
        "staged_bytes": total,
        "text_files_scanned": text_count,
        "checks": list(PATTERNS) + ["credential_from_environment", "excluded_file_types", "blob_size"],
        "findings": findings,
        "passed": not findings,
        "directories": {key: {"files": counts[key], "bytes": sizes[key]} for key in sorted(counts)},
        "largest_files": [{"path": path, "bytes": length}
                          for length, path in sorted(largest, reverse=True)[:10]],
        "limitations": "Pattern-based credential checks cannot recognize every possible secret.",
    }
    destination = root / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "files": len(entries),
                      "bytes": total, "findings": findings}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
