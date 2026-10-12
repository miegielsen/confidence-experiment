"""Pin the original dataset and persist five disjoint data roles."""
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from .io import read_json, write_json

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
ROLES = ("fit", "correctness", "calibration", "development", "test")


def prepare(directory, seed=42):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["seed"] != seed:
            raise ValueError("Existing data uses another seed. Choose a new data_dir.")
        load(directory)
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    hashes, frames = {}, {}
    for split in ("train", "test"):
        payload = urlopen(f"{BASE}/{split}.csv", timeout=60).read()
        (directory / f"official_{split}.csv").write_bytes(payload)
        hashes[split] = hashlib.sha256(payload).hexdigest()
        frame = pd.read_csv(io.BytesIO(payload)).rename(columns={"category": "label"})
        frame["id"] = [f"{split}:{i}" for i in range(len(frame))]
        frame["text_key"] = frame.text.str.casefold().str.split().str.join(" ")
        frames[split] = frame
    train, test = frames["train"], frames["test"]
    labels = sorted(train.label.unique().tolist())
    if len(labels) != 77 or len(train) != 10003 or len(test) != 3080:
        raise ValueError("Unexpected BANKING77 contents; inspect the pinned source.")
    # Quarantine conflicting duplicate training texts, then deduplicate the rest.
    conflicts = train.groupby("text_key").label.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    overlap_keys = set(train.text_key) & set(test.text_key)
    eligible = train[~train.text_key.isin(conflict_keys | overlap_keys)]
    clean = eligible.drop_duplicates("text_key").copy()
    fit, rest = train_test_split(clean, test_size=4000, stratify=clean.label, random_state=seed)
    correctness, rest = train_test_split(rest, test_size=2000, stratify=rest.label, random_state=seed + 1)
    calibration, development = train_test_split(rest, test_size=1000, stratify=rest.label, random_state=seed + 2)
    roles = dict(zip(ROLES, (fit, correctness, calibration, development, test)))
    split_hashes = {}
    for role, frame in roles.items():
        target = directory / f"{role}.csv"
        frame[["id", "text", "label"]].sort_values("id").to_csv(target, index=False)
        split_hashes[role] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "dataset": "BANKING77", "revision": REVISION, "seed": seed, "labels": labels,
        "official_sha256": hashes, "split_sha256": split_hashes,
        "sizes": {r: len(f) for r, f in roles.items()},
        "audit": {"conflicting_train_texts": len(conflict_keys), "train_test_overlap_texts": len(overlap_keys),
                  "duplicate_train_rows_removed": len(eligible) - len(clean),
                  "total_train_rows_excluded": len(train) - len(clean),
                  "duplicate_test_texts": int(test.text_key.duplicated().sum())},
        "policy": "Remove conflicting training texts, remove train/test text overlaps from training, deduplicate training. Preserve official test rows and labels.",
    }
    write_json(manifest_path, manifest)
    return manifest


def load(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    frames = {}
    for role in ROLES:
        path = directory / f"{role}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["split_sha256"][role]:
            raise ValueError(f"Dataset split changed: {role}")
        frames[role] = pd.read_csv(path, keep_default_na=False)
    return manifest, frames
