#!/usr/bin/env python3
"""Generate one ScribbleBraTS annotation round from 2-D BraTS PNG masks."""

import argparse
from functools import partial
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
from skimage import morphology
from tqdm import tqdm

from simulationcodes.utils import (
    decide_path_or_boundary_medical,
    get_bezier_points,
    get_dfs_points,
    get_random_boundary_v2,
    xtract_minEnclosingCircle,
    xtract_region_by_circle,
)


def sample_seed(seed: int, name: str) -> int:
    return (seed + sum((index + 1) * ord(char) for index, char in enumerate(name))) % (2**32 - 1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, regions, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (regions == component).astype(np.uint8)


def foreground_path(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if mask.sum() <= 50 or not decide_path_or_boundary_medical(mask):
        skeleton = morphology.medial_axis(mask, rng=rng).astype(np.int32)
        return np.asarray(get_dfs_points(skeleton))
    try:
        return np.asarray(get_random_boundary_v2(mask, jitter_pixel=3))
    except (IndexError, ValueError, cv2.error):
        skeleton = morphology.medial_axis(mask, rng=rng).astype(np.int32)
        return np.asarray(get_dfs_points(skeleton))


def generate_one(label_path: Path, image_dir: Path, output_dir: Path, seed: int) -> str:
    basename = label_path.name.removesuffix("_seg.png")
    deterministic_seed = sample_seed(seed, basename)
    np.random.seed(deterministic_seed)
    rng = np.random.default_rng(deterministic_seed)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_dir / f"{basename}_t1ce.png"), cv2.IMREAD_COLOR)
    if label is None or image is None:
        raise FileNotFoundError(f"missing label or T1ce image for {basename}")
    if label.shape != image.shape[:2]:
        raise ValueError(f"{basename}: image and label dimensions differ")
    labels = set(np.unique(label).tolist())
    if not labels.issubset({0, 1, 2, 3}):
        raise ValueError(f"{basename}: expected labels 0/1/2/3, found {sorted(labels)}")

    if not np.any(label):
        cv2.imwrite(str(output_dir / f"{basename}.png"), np.zeros_like(label))
        return basename

    scribble = np.full_like(label, 255)
    retained_foreground = np.zeros_like(label, dtype=np.uint8)
    for class_id in (1, 2, 3):
        class_mask = (label == class_id).astype(np.uint8)
        if class_mask.sum() < 10:
            continue
        component = largest_component(class_mask)
        height, width = component.shape
        component = cv2.rectangle(component, (1, 1), (width - 1, height - 1), 0, thickness=3)
        if not component.any():
            continue
        path = foreground_path(component, rng).astype(np.int32)
        if len(path) >= 2:
            cv2.polylines(scribble, [path], False, int(class_id), thickness=1)
        retained_foreground[component > 0] = 1

    if retained_foreground.any():
        center, radius = xtract_minEnclosingCircle(retained_foreground.copy())
        prior = xtract_region_by_circle(image, center, int(radius * 5), return_binary=False)
        background = ((label == 0)[:, :, None] & prior).astype(np.uint8)
        height, width = label.shape
        background = cv2.rectangle(background, (1, 1), (width - 1, height - 1), (0, 0, 0), thickness=3)
        skeleton, distance = morphology.medial_axis(background[:, :, 0], return_distance=True, rng=rng)
        weighted_skeleton = (distance * skeleton).astype(np.uint8)
        clean_skeleton = largest_component((weighted_skeleton > 0).astype(np.uint8))
        if clean_skeleton.any():
            path = np.asarray(get_dfs_points(clean_skeleton.astype(np.int32)))
            if len(path) >= 5:
                path = get_bezier_points(path)[2:-2].astype(np.int32)
                if len(path) >= 2:
                    cv2.polylines(scribble, [path], False, 0, thickness=1)
    cv2.imwrite(str(output_dir / f"{basename}.png"), scribble)
    return basename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Preprocessed images/ directory containing *_t1ce.png")
    parser.add_argument("--label-dir", type=Path, required=True, help="Preprocessed GT/ directory containing *_seg.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination scribble_rN directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407, help="Random seed used in the paper")
    args = parser.parse_args()

    labels = sorted(args.label_dir.glob("*_seg.png"))
    if not labels:
        raise ValueError(f"no *_seg.png files found in {args.label_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = partial(generate_one, image_dir=args.image_dir, output_dir=args.output_dir, seed=args.seed)
    with mp.Pool(args.workers) as pool:
        list(tqdm(pool.imap(worker, labels), total=len(labels), desc="Generating scribbles"))


if __name__ == "__main__":
    main()
