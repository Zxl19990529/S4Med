# Dataset preparation and fixed splits

This directory contains ScribbleBraTS construction tools and the frozen patient-level train/validation splits for both ScribbleBraTS and ScribbleMyoPS.

Run the commands below from `MakeScribbleMedicalDatasets/`.

## Fixed train/validation splits

```text
ImageSets/
  ScribbleBraTS/SegmentationAug/{train,val}.txt
  ScribbleMyoPS/SegmentationAug/{train,val}.txt
```

| Dataset | Training patients | Validation patients | Training slices | Validation slices |
| --- | --- | --- | --- | --- |
| ScribbleBraTS | 875 | 376 | 121,785 | 52,304 |
| ScribbleMyoPS | 140 | 60 | 577 | 239 |

The patient sets do not overlap between training and validation. To install the paper splits in your dataset directories:

```bash
mkdir -p /path/to/ScribbleBraTS/ImageSets /path/to/ScribbleMyoPS/ImageSets
cp -r ImageSets/ScribbleBraTS/SegmentationAug /path/to/ScribbleBraTS/ImageSets/
cp -r ImageSets/ScribbleMyoPS/SegmentationAug /path/to/ScribbleMyoPS/ImageSets/
```

Each dataset loader reads `<dataset_root>/ImageSets/SegmentationAug/train.txt` and `val.txt`.

## ScribbleBraTS preparation

Install the dependencies in a clean Python environment. The preparation code was checked with Python 3.11 and scikit-image 0.25.2:

```bash
python -m pip install -r requirements.txt
```

The input directory must contain one case directory per patient. Each case needs exactly one NIfTI file ending in `_t1n.nii.gz`, `_t1c.nii.gz`, `_t2w.nii.gz`, `_t2f.nii.gz`, and `_seg.nii.gz`.

```bash
python preprocess_brats2023.py \
  --input-dir /path/to/BraTS2023-GLI-TrainingData \
  --output-dir /path/to/ScribbleBraTS \
  --workers 8
```

The preprocessor creates `images/` and `GT/`. Each retained slice has the basename `<patient>_sliceNNN`, with four images named `<basename>_{t1,t1ce,t2,flair}.png` and a dense label named `GT/<basename>_seg.png`.

The NIfTI labels are written using the original S4Med PNG mapping: `0` background, `1` Edema Tumor (ED), `2` Non-Enhanced Tumor (NET), and `3` Enhanced Tumor (ET). This swaps the source NIfTI values `1` and `2` to preserve compatibility with the released network and evaluation code. Slices whose four MRI modalities are all zero are skipped by default, matching the original preparation workflow; use `--keep-empty` to retain them.

Generate four annotation rounds into the exact layout consumed by `ScribbleMedicalCodes/dataset.py`:

```bash
python generate_brats_scribbles.py --image-dir /path/to/ScribbleBraTS/images --label-dir /path/to/ScribbleBraTS/GT --output-dir /path/to/ScribbleBraTS/scribble_r1 --workers 8 --seed 3407
python generate_brats_scribbles.py --image-dir /path/to/ScribbleBraTS/images --label-dir /path/to/ScribbleBraTS/GT --output-dir /path/to/ScribbleBraTS/scribble_r2 --workers 8 --seed 3408
python generate_brats_scribbles.py --image-dir /path/to/ScribbleBraTS/images --label-dir /path/to/ScribbleBraTS/GT --output-dir /path/to/ScribbleBraTS/scribble_r3 --workers 8 --seed 3409
python generate_brats_scribbles.py --image-dir /path/to/ScribbleBraTS/images --label-dir /path/to/ScribbleBraTS/GT --output-dir /path/to/ScribbleBraTS/scribble_r4 --workers 8 --seed 3410
```

Each scribble PNG uses `0`, `1`, `2`, and `3` for annotated background, Edema Tumor (ED), Non-Enhanced Tumor (NET), and Enhanced Tumor (ET) pixels respectively; all unannotated pixels are `255` and must be passed to a loss with ignore index `255`. A pure-background slice is stored as all-zero because its background is fully annotated.

Install the frozen ScribbleBraTS split described above after preprocessing. `split_brats_patients.py` is provided only to make a new patient-level split when necessary:

```bash
python split_brats_patients.py --label-dir /path/to/ScribbleBraTS/GT --output-dir /path/to/ScribbleBraTS/ImageSets/SegmentationAug --train-ratio 0.7 --seed 3407
```

The generator implements the original structure-aware skeleton/boundary simulation, including its stochastic path/boundary decision threshold (`0.3333`). Its default seed is `3407`, the paper seed. A deterministic seed is derived independently for each slice, so a given seed yields the same result with different worker counts. This makes the release reproducible; the legacy script had no exposed seed and could vary with process scheduling. The four example rounds use consecutive seeds; use the frozen dataset release for the exact published annotation files.

Dense pseudo-labels are not generated in this repository. For that stage, use the official [ScribblePrompt repository](https://github.com/halleewong/ScribblePrompt) and its license.

Download the prepared ScribbleBraTS dataset from the [repository download table](../README.md#datasets). To rebuild it from NIfTI files, obtain source images under the original BraTS access terms.
