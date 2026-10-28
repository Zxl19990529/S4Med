#!/usr/bin/env python3
"""Convert BraTS 2023 GLI NIfTI cases into the 2-D PNG layout used by S4Med."""

import argparse
from functools import partial
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


MODALITY_SUFFIXES = {
    "flair": "t2f",
    "t1": "t1n",
    "t2": "t2w",
    "t1ce": "t1c",
    "seg": "seg",
}


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = image.min(), image.max()
    return ((image - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def encode_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Apply the legacy BraTS PNG mapping used by the S4Med loader and metrics."""
    labels = set(np.unique(segmentation).tolist())
    if not labels.issubset({0, 1, 2, 3}):
        raise ValueError(f"expected BraTS-2023 labels 0/1/2/3, found {sorted(labels)}")
    encoded = np.zeros_like(segmentation, dtype=np.uint8)
    encoded[segmentation == 1] = 2
    encoded[segmentation == 2] = 1
    encoded[segmentation == 3] = 3
    return encoded


def find_case_files(case_dir: Path) -> dict[str, Path]:
    files = {}
    for modality, suffix in MODALITY_SUFFIXES.items():
        matches = sorted(case_dir.glob(f"*_{suffix}.nii.gz"))
        if len(matches) != 1:
            raise FileNotFoundError(f"{case_dir.name}: expected one *_{suffix}.nii.gz, found {len(matches)}")
        files[modality] = matches[0]
    return files


def process_case(case_dir: Path, output_dir: Path, skip_empty: bool) -> str:
    files = find_case_files(case_dir)
    volumes = {name: sitk.GetArrayFromImage(sitk.ReadImage(path)) for name, path in files.items()}
    reference_shape = volumes["flair"].shape
    if any(volume.shape != reference_shape for volume in volumes.values()):
        raise ValueError(f"{case_dir.name}: modality volumes do not share a shape")

    images_dir = output_dir / "images"
    gt_dir = output_dir / "GT"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for slice_index in range(reference_shape[0]):
        image_slices = {name: volumes[name][slice_index] for name in ("t1", "t1ce", "t2", "flair")}
        if skip_empty and not any(np.count_nonzero(image) for image in image_slices.values()):
            continue
        basename = f"{case_dir.name}_slice{slice_index:03d}"
        for name, image in image_slices.items():
            cv2.imwrite(str(images_dir / f"{basename}_{name}.png"), normalize_image(image))
        cv2.imwrite(str(gt_dir / f"{basename}_seg.png"), encode_segmentation(volumes["seg"][slice_index]))
    return case_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing one BraTS case directory per patient")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination dataset directory; writes images/ and GT/")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-empty", action="store_true", help="Also retain slices for which every MRI modality is zero")
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"no case directories found in {args.input_dir}")
    worker = partial(process_case, output_dir=args.output_dir, skip_empty=not args.keep_empty)
    with mp.Pool(args.workers) as pool:
        list(tqdm(pool.imap(worker, case_dirs), total=len(case_dirs), desc="Preprocessing cases"))


if __name__ == "__main__":
    main()
