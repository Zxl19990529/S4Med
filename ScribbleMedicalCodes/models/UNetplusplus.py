import math
import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


def upsample_like(x, ref):
	x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
	diffY = ref.size()[2] - x.size()[2]
	diffX = ref.size()[3] - x.size()[3]
	x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
	return x


class UNetplusplus(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=False):
		super(UNetplusplus, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		filters = [64, 128, 256, 512, 1024]

		self.conv0_0 = DoubleConv(n_channels, filters[0])
		self.conv1_0 = Down(filters[0], filters[1])
		self.conv2_0 = Down(filters[1], filters[2])
		self.conv3_0 = Down(filters[2], filters[3])
		self.conv4_0 = Down(filters[3], filters[4])

		self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
		self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
		self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
		self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3])

		self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
		self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
		self.conv2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2])

		self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
		self.conv1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1])

		self.conv0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0])

		self.outc = OutConv(filters[0], n_classes)

	def forward(self, x):
		x0_0 = self.conv0_0(x)
		x1_0 = self.conv1_0(x0_0)
		x2_0 = self.conv2_0(x1_0)
		x3_0 = self.conv3_0(x2_0)
		x4_0 = self.conv4_0(x3_0)

		x0_1 = self.conv0_1(torch.cat([x0_0, upsample_like(x1_0, x0_0)], dim=1))
		x1_1 = self.conv1_1(torch.cat([x1_0, upsample_like(x2_0, x1_0)], dim=1))
		x2_1 = self.conv2_1(torch.cat([x2_0, upsample_like(x3_0, x2_0)], dim=1))
		x3_1 = self.conv3_1(torch.cat([x3_0, upsample_like(x4_0, x3_0)], dim=1))

		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_like(x1_1, x0_0)], dim=1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_like(x2_1, x1_0)], dim=1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_like(x3_1, x2_0)], dim=1))

		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_like(x1_2, x0_0)], dim=1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_like(x2_2, x1_0)], dim=1))

		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_like(x1_3, x0_0)], dim=1))

		logits = self.outc(x0_4)
		return logits, x0_4

	def forward_eval(self, x):
		x0_0 = self.conv0_0(x)
		x1_0 = self.conv1_0(x0_0)
		x2_0 = self.conv2_0(x1_0)
		x3_0 = self.conv3_0(x2_0)
		x4_0 = self.conv4_0(x3_0)

		x0_1 = self.conv0_1(torch.cat([x0_0, upsample_like(x1_0, x0_0)], dim=1))
		x1_1 = self.conv1_1(torch.cat([x1_0, upsample_like(x2_0, x1_0)], dim=1))
		x2_1 = self.conv2_1(torch.cat([x2_0, upsample_like(x3_0, x2_0)], dim=1))
		x3_1 = self.conv3_1(torch.cat([x3_0, upsample_like(x4_0, x3_0)], dim=1))

		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_like(x1_1, x0_0)], dim=1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_like(x2_1, x1_0)], dim=1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_like(x3_1, x2_0)], dim=1))

		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_like(x1_2, x0_0)], dim=1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_like(x2_2, x1_0)], dim=1))

		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_like(x1_3, x0_0)], dim=1))

		logits = self.outc(x0_4)
		return logits


UNetPlusPlus = UNetplusplus
