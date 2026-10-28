# S4Med segmentation code

This directory contains the training and evaluation code for **S4M$^2$RI**, the segmentation framework in *Scribble-Supervised Semantic Segmentation for Multi-sequence Magnetic Resonance Images*.  The repository is named S4Med; the method name in the paper is S4M$^2$RI.

S4M$^2$RI trains a multi-sequence segmentation model with sparse scribble supervision, sequence-specific dense pseudo-label supervision, and cascaded tree-filter regularization.  The release includes the four backbones used in the paper: r101v3p (DeepLabV3+ with ResNet-101), UNet, UNet++, and ResUNet.

## Setup

Run the commands below from `ScribbleMedicalCodes/`.

Create a Python environment with PyTorch and a CUDA toolkit compatible with the installed PyTorch build, then install the Python dependencies:

```bash
pip install -r requirements.txt
cd kernels/lib_tree_filter
python setup.py build_ext --inplace
cd ../..
```

The tree-filter extension is required when `--l_tree` is nonzero.  No compiled binaries, datasets, pseudo-labels, or checkpoints are included in this repository.

## Data layout

Put a released dataset under `data/` or pass its path with `--dataset_path`.

```text
data/ScribbleBraTS/
  images/{slice}_t1.png
  images/{slice}_t1ce.png
  images/{slice}_t2.png
  images/{slice}_flair.png
  GT/{slice}_seg.png
  scribble_r1/{slice}.png ... scribble_r4/{slice}.png
  ImageSets/SegmentationAug/{train,val}.txt

data/ScribbleMyoPS/
  images/{slice}_C0.png
  images/{slice}_T2.png
  images/{slice}_DE.png
  GT/{slice}_gd.png
  scribble_r1/{slice}.png ... scribble_r3/{slice}.png
  ImageSets/SegmentationAug/{train,val}.txt
```

Pseudo-label directories are generated separately with the official [ScribblePrompt repository](https://github.com/halleewong/ScribblePrompt).  They are not generated or redistributed here.  For BraTS, the expected directories are `t1_vit_l_pseudo`, `t1ce_vit_l_pseudo`, `t2_vit_l_pseudo`, and `flair_vit_l_pseudo`.  For MyoPS, they are `C0_vit_l_pseudo`, `DE_vit_l_pseudo`, and `T2_vit_l_pseudo`.

These directories live under the dataset root. Each contains `<slice>_<sequence>.png`, for example `t1ce_vit_l_pseudo/<slice>_t1ce.png` or `DE_vit_l_pseudo/<slice>_DE.png`, with the same class values as the dense reference mask.

BraTS uses background, Non-Enhanced Tumor (NET), Edema Tumor (ED), and Enhanced Tumor (ET) labels.  The released PNG convention is ED=1, NET=2, and ET=3; the BraTS regional evaluator therefore computes ET from label 3, TC from labels 2 and 3, and WT from labels 1, 2, and 3.  ScribbleMyoPS uses background plus LV normal myocardium (LVn), LV blood pool (LVb), RV blood pool (RVb), LV edema (LVe), and LV scar (LVs).  The metric scripts exclude a class from a slice-level summary when both prediction and reference are empty for that class.

The evaluators score the supplied prediction PNGs, require a matching ground-truth PNG for every prediction, and stop on unreadable files; they do not silently score a subset.

## Reproducing the main configuration

The paper uses SGD (momentum 0.9, weight decay 5e-4), an initial learning rate of 5e-3, batch size 32, five warm-up epochs, seed 3407, and tree-filter scales `gamma_low=0.02` and `gamma_high=1.0`.  These tree-filter scales are the defaults in `loss.py`.

For ScribbleBraTS, use 20 epochs, label smoothing 0.1, and weights 0.1 for the tree loss and each sequence pseudo-label loss:

```bash
python trainBraTS.py --dataset_path data/ScribbleBraTS --dataset ScribblePseudo --model_type deeplabv3p --layers 101 --numclasses 4 --modality 0 1 2 3 --epochs 20 --batchsize 32 --lr 5e-3 --label_smooth 0.1 --l_segs 1 --l_tree 0.1 --l_segp_1 0.1 --l_segp_2 0.1 --l_segp_3 0.1 --l_segp_4 0.1 --logdir outputs/ScribbleBraTS
```

For ScribbleMyoPS, use 50 epochs, label smoothing 0.1, and weights 0.4 for the tree loss and each sequence pseudo-label loss:

```bash
python trainMyoPS.py --dataset_path data/ScribbleMyoPS --dataset ScribblePseudo --model_type deeplabv3p --layers 101 --numclasses 6 --modality 0 1 2 --epochs 50 --batchsize 32 --lr 5e-3 --label_smooth 0.1 --l_segs 1 --l_tree 0.4 --l_segp_1 0.4 --l_segp_2 0.4 --l_segp_3 0.4 --logdir outputs/ScribbleMyoPS
```

To train with scribbles only, set `--dataset ScribbleOnly` and leave `--l_tree` and all `--l_segp_*` values at zero.

## Evaluation

Run prediction generation with a saved checkpoint, then compute slice-level metrics.  The metric scripts report Dice, HD95, and sensitivity across valid validation slices.

```bash
python eval_BraTS.py --dataset_path data/ScribbleBraTS --model_type deeplabv3p --layers 101 --numclasses 4 --modality 0 1 2 3 --checkpoint_path outputs/ScribbleBraTS/RUN/last_checkpoint.pth --save_path outputs/eval_ScribbleBraTS --colormap BraTS
python eval_masksBraTS_region.py --gt_folder data/ScribbleBraTS/GT --mask_folder outputs/eval_ScribbleBraTS/GT --csv_path outputs/eval_ScribbleBraTS/metrics.csv
```

For ScribbleMyoPS:

```bash
python eval_MyoPS.py --dataset_path data/ScribbleMyoPS --model_type deeplabv3p --layers 101 --numclasses 6 --modality 0 1 2 --checkpoint_path outputs/ScribbleMyoPS/RUN/last_checkpoint.pth --save_path outputs/eval_ScribbleMyoPS --colormap MyoPS
python eval_masksMyoPS.py --gt_folder data/ScribbleMyoPS/GT --mask_folder outputs/eval_ScribbleMyoPS/GT --csv_path outputs/eval_ScribbleMyoPS/metrics.csv
```

## Notices

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for included third-party components.
