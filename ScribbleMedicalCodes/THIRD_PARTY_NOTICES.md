# Third-party notices

`models/UNet.py` was adapted from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), released under GPL-3.0. See [`../LICENSES/Pytorch-UNet-GPL-3.0.txt`](../LICENSES/Pytorch-UNet-GPL-3.0.txt).
This release retains its S4Med-specific multi-sequence interface, which returns both logits and decoder features.

`sync_batchnorm/` was adapted from [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch), released under the MIT License. See [`../LICENSES/Synchronized-BatchNorm-MIT.txt`](../LICENSES/Synchronized-BatchNorm-MIT.txt).

The CUDA implementation under `kernels/lib_tree_filter/` is derived from TreeEnergyLoss and is licensed under Apache-2.0. See [`../LICENSES/TreeEnergyLoss-Apache-2.0.txt`](../LICENSES/TreeEnergyLoss-Apache-2.0.txt). It must be built locally; no compiled extension is included.
The included tree-filter wrapper exposes separate low- and high-level affinity scales used by S4M$^2$RI.
