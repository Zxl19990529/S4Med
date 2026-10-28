import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
import torchnet as tnt
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as P
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from torch.utils import data
try:
    from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
    from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D
    from kernels.lib_tree_filter.functions.bfs import bfs
except Exception as e:
    print(e)
# label = 255 is ambiguious label, and only some gts have this label.
class SegLoss ( nn.Module ) :
    def __init__(self, ignore_label=255, mode=1,smooth=0) :
        super ( SegLoss, self ).__init__ ()
        self.obj = torch.nn.CrossEntropyLoss ( ignore_index=ignore_label ,label_smoothing=smooth)

    def __call__(self, pred, label) :
        loss = self.obj ( pred, label )
        return loss


class EntropyLoss ( nn.Module ) :
    def __init__(self) :
        super ( EntropyLoss, self ).__init__ ()

    def forward(self, x, mask, mode=1) : # mask : superpixel
        # mask_size = mask.size()[1:3]
        # x_softmax = F.softmax(x, dim = 1)
        # x_logsoftmax = F.log_softmax(x, dim = 1)
        # x_softmax_up = F.interpolate(x_softmax, size=mask_size, mode='bilinear', align_corners=True)
        # x_logsoftmax_up = F.interpolate(x_logsoftmax, size=mask_size, mode='bilinear', align_corners=True)
        # b = x_softmax_up * x_logsoftmax_up

        if mode == 1 :
            mask = 1.0 - mask / 255 # 除去superpixel位置，其余全是1  【4，465，465】
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 ) # (4,21,465,465)
            b = torch.sum ( b, dim=1 )  # (4,465,465)
            entropy = b.mul ( mask ) # 4,465,465
            loss = -1.0 * torch.sum ( entropy ) / torch.sum ( mask )
        else :
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            loss = -1.0 * torch.mean ( b )
        return loss

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Supports multi-class segmentation
    """

    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) - model predictions (logits)
            target: (N, H, W) - ground truth labels
        Returns:
            dice_loss: scalar
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)

        # Handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask.unsqueeze(1)
            target = target * mask

        # Convert target to one-hot encoding
        num_classes = pred.size(1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Calculate intersection and union
        intersection = torch.sum(pred * target_onehot, dim=(2, 3))
        union = torch.sum(pred + target_onehot, dim=(2, 3))

        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        dice_loss = 1 - dice

        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

def save_normalized_tensor_as_image(tensor, points=None, output_path="tmp.png", point_color="red", point_radius=3):
    """
    将归一化的 torch.tensor (c, h, w) 转换为 0-255 的 RGB 图像并保存。
    可选：可视化点集 points (N, 2)，每点为 (x, y) 坐标。
    """
    # 转换 tensor 到图像
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    image_np = tensor.permute(1, 2, 0).numpy()
    image_pil = Image.fromarray(image_np)

    # 可视化点
    if points is not None:
        draw = ImageDraw.Draw(image_pil)
        points = np.array(points)  # 确保是 numpy 数组
        for x, y in points:
            draw.ellipse([x-point_radius, y-point_radius, x+point_radius, y+point_radius], fill=point_color)

    image_pil.save(output_path)
    print(f"Image with points saved to {output_path}")

## new
class TreeEnergyLoss(nn.Module):
    def __init__(self, gamma_low=0.02, gamma_high=1.0, configer=None):
        super(TreeEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(
            groups=1,
            gamma_low=gamma_low,
            gamma_high=gamma_high,
        )

    def _prepare_inputs(self, preds, low_feats, high_feats, unlabeled_ROIs):
        batch, _, h, w = preds.size()
        low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
        if high_feats is not None and high_feats.shape[-2:] != (h, w):
            high_feats = F.interpolate(high_feats, size=(h, w), mode='bilinear', align_corners=False)
        unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
        return batch, h, w, low_feats, high_feats, unlabeled_ROIs

    def collect_tree_debug(self, preds, low_feats, high_feats, unlabeled_ROIs):
        with torch.no_grad():
            _, _, _, low_feats, high_feats, unlabeled_ROIs = self._prepare_inputs(
                preds, low_feats, high_feats, unlabeled_ROIs
            )

        prob = torch.softmax(preds, dim=1)

        low_tree = self.mst_layers(low_feats)
        low_sorted_index, low_sorted_parent, low_sorted_child = bfs(low_tree, 4)
        low_edge_weight = self.tree_filter_layers.build_edge_weight(
            low_feats, low_sorted_index, low_sorted_parent, low_tree=True
        )
        low_as = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=low_tree)

        debug_info = {
            'prob': prob,
            'low_feats': low_feats,
            'unlabeled_rois': unlabeled_ROIs,
            'low_tree': low_tree,
            'low_sorted_index': low_sorted_index,
            'low_sorted_parent': low_sorted_parent,
            'low_sorted_child': low_sorted_child,
            'low_edge_weight': low_edge_weight,
            'low_as': low_as,
        }

        if high_feats is not None:
            high_tree = self.mst_layers(high_feats)
            high_sorted_index, high_sorted_parent, high_sorted_child = bfs(high_tree, 4)
            high_edge_weight = self.tree_filter_layers.build_edge_weight(
                high_feats, high_sorted_index, high_sorted_parent, low_tree=False
            )
            high_as = self.tree_filter_layers(
                feature_in=low_as, embed_in=high_feats, tree=high_tree, low_tree=False
            )
            debug_info.update({
                'high_feats': high_feats,
                'high_tree': high_tree,
                'high_sorted_index': high_sorted_index,
                'high_sorted_parent': high_sorted_parent,
                'high_sorted_child': high_sorted_child,
                'high_edge_weight': high_edge_weight,
                'high_as': high_as,
            })
        else:
            debug_info['high_feats'] = None
            debug_info['high_as'] = None

        return debug_info

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs): # preds: direct logits, low_feats:original image high_feats: high-level features from embedding layers, unlabeled_ROIs: mask for unlabeled pixels
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, h, w, low_feats, high_feats, unlabeled_ROIs = self._prepare_inputs(
                preds, low_feats, high_feats, unlabeled_ROIs
            )
            N = unlabeled_ROIs.sum()

        prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats) # b,6560,2
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return tree_loss
