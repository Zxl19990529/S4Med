# S4Med

Code and dataset preparation tools for **Scribble-Supervised Semantic Segmentation for Multi-sequence Magnetic Resonance Images**. The framework is named **S4M²RI** in the paper.

S4M²RI combines sequence-specific dense pseudo-label supervision with structure-aware graph regularization for scribble-supervised MRI segmentation.

## Code

| Directory | Contents |
| --- | --- |
| [MakeScribbleMedicalDatasets](MakeScribbleMedicalDatasets/) | BraTS preprocessing, structure-aware scribble simulation, and dataset preparation |
| [ScribbleMedicalCodes](ScribbleMedicalCodes/) | Segmentation models, training, inference, and slice-level evaluation for ScribbleBraTS and ScribbleMyoPS |

Each directory has its own installation instructions and runnable examples. Start with the [dataset preparation guide](MakeScribbleMedicalDatasets/README.md), then follow the [training and evaluation guide](ScribbleMedicalCodes/README.md).

## Datasets

| Dataset | MRI sequences | Foreground classes |
| --- | --- | --- |
| ScribbleBraTS | T1, T1ce, T2, FLAIR | Enhanced Tumor (ET), Edema Tumor (ED), Non-Enhanced Tumor (NET) |
| ScribbleMyoPS | C0, DE, T2 | Normal LV (LVn), LV blood pool (LVb), RV blood pool (RVb), LV edema (LVe), LV scar (LVs) |

Scribble labels use `0` for background and `255` for unannotated pixels. Class values and filenames are documented in the component guides. BraTS evaluation reports the nested ET, TC, and WT regions derived from the primitive classes.

Download the prepared datasets from Baidu Netdisk:

| Dataset | Archive | Download | Extraction code |
| --- | --- | --- | --- |
| ScribbleBraTS | ScribbleBraTS-JBHI | [Baidu Netdisk](https://pan.baidu.com/s/1h_Ga9Cl8ysdsK4BbuLMJYg?pwd=sfgw) | `sfgw` |
| ScribbleMyoPS | ScribbleMyoPS-JBHI | [Baidu Netdisk](https://pan.baidu.com/s/1ihH0JCe4elCOevjd_pYjGg?pwd=7g4i) | `7g4i` |

Extract the corresponding archive and use the directory containing `images/` and `GT/` as the dataset root. Dataset archives are distributed separately from the source code.

The fixed train/validation lists are included for [ScribbleBraTS](MakeScribbleMedicalDatasets/ImageSets/ScribbleBraTS/SegmentationAug/) and [ScribbleMyoPS](MakeScribbleMedicalDatasets/ImageSets/ScribbleMyoPS/SegmentationAug/). Copy each dataset's `SegmentationAug/` directory into its dataset root's `ImageSets/` directory, following the [split installation guide](MakeScribbleMedicalDatasets/README.md#fixed-trainvalidation-splits).

## Pseudo-labels

We generated the sequence-specific dense pseudo-labels using the [official ScribblePrompt repository](https://github.com/halleewong/ScribblePrompt). Use its official implementation and pretrained model for pseudo-label generation. The expected pseudo-label filenames and directories are described in the [segmentation guide](ScribbleMedicalCodes/README.md#data-layout).

## Training settings

The paper environment uses Python 3.11, PyTorch 2.9.1, torchvision 0.24.1, and CUDA 12.8.

The paper uses SGD with momentum `0.9`, weight decay `5e-4`, initial learning rate `5e-3`, batch size `32`, five warmup epochs, cosine learning-rate decay, and pseudo-label smoothing `0.1`. The tree affinity scales are `gamma_low=0.02` and `gamma_high=1.0`.

| Dataset | Epochs | Graph loss weight | Dense pseudo-label loss weight |
| --- | --- | --- | --- |
| ScribbleBraTS | 20 | 0.1 | 0.1 |
| ScribbleMyoPS | 50 | 0.4 | 0.4 |

Use a patient-level train/validation partition, with all slices from one patient in the same partition. Evaluation computes Dice, HD95, and sensitivity per validation slice and reports mean ± standard deviation over valid values. HD95 is measured in pixels.

## License

Original S4Med code and contributions are licensed under the [MIT License](LICENSE). Third-party implementations retain the licenses listed in the [third-party notices](ScribbleMedicalCodes/THIRD_PARTY_NOTICES.md).

The adapted UNet implementation is licensed under GPL-3.0. Distribution of combined works incorporating that implementation is subject to the GPL-3.0 requirements.

The code license applies to software. Dataset use follows the original BraTS and MyoPS dataset terms.

## Acknowledgments

We thank the authors of ScribblePrompt and the upstream model and tree-filter implementations. Third-party source notices and license texts are retained with the corresponding components.
