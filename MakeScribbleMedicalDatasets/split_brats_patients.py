#!/usr/bin/env python3
"""Create patient-level train/validation lists for the S4Med PNG layout."""

import argparse
from pathlib import Path
import random
import re


NAME = re.compile(r"^(?P<patient>.+)_slice(?P<slice>\d+)$")


def collect(label_dir: Path) -> dict[str, list[str]]:
    patients: dict[str, list[str]] = {}
    for path in sorted(label_dir.glob("*_seg.png")):
        stem = path.name.removesuffix("_seg.png")
        match = NAME.match(stem)
        if not match:
            raise ValueError(f"unexpected label filename: {path.name}")
        patients.setdefault(match.group("patient"), []).append(stem)
    for names in patients.values():
        names.sort(key=lambda name: int(NAME.match(name).group("slice")))
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True, help="GT/ directory containing *_seg.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination ImageSets/SegmentationAug directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be strictly between 0 and 1")
    patients = collect(args.label_dir)
    if not patients:
        raise ValueError(f"no labels found in {args.label_dir}")
    patient_names = sorted(patients)
    random.Random(args.seed).shuffle(patient_names)
    count = int(len(patient_names) * args.train_ratio)
    train_patients = set(patient_names[:count])
    train = [name for patient in sorted(train_patients) for name in patients[patient]]
    validation = [name for patient in sorted(set(patient_names) - train_patients) for name in patients[patient]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.output_dir / "val.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    print(f"patients: {len(patient_names)}; train/val: {len(train_patients)}/{len(patient_names) - len(train_patients)}")
    print(f"slices: {len(train)}/{len(validation)}")


if __name__ == "__main__":
    main()
