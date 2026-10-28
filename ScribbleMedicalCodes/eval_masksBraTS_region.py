import os
import csv
import cv2
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import cv2
from medpy.metric.binary import dc,sensitivity,specificity,hd95
from functools import partial
from multiprocessing import Pool, cpu_count


REGION_NAMES = ["ET", "TC", "WT"]


def color_to_class_mask(img, color_list):
    """
    将彩色图像中的指定颜色映射为指定类别（类别用颜色表示），输出三通道mask。
    :param img: 输入三通道彩色图像（H, W, 3），np.uint8
    :param color_list: 颜色列表，每个元素为 (R, G, B) 元组
    :return: 三通道mask（H, W, 3），每个像素为对应类别的颜色
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        # 创建与原图同样大小的布尔掩码
        match = np.all(img == color, axis=-1)
        mask[match] = idx  # 或者 mask[match] = color_list[idx]
    return mask


def dice_coefficient(mask, gt, num_classes):
    """
    使用medpy库计算每个类别的DSC
    :param mask: 预测mask，二维数组
    :param gt: GT mask，二维数组
    :param num_classes: 类别数
    :return: 每个类别的DSC列表
    """
    dsc_list = []
    for cls in range(num_classes):
        mask_cls = (mask == cls)
        gt_cls = (gt == cls)
        # medpy的dc函数要求输入为bool类型
        if np.sum(gt_cls) == 0 and np.sum(mask_cls) == 0:
            dsc = np.nan  # 如果GT和预测都没有该类，DSC定义为1
            dsc_list.append(dsc)
            continue
        if np.sum(mask_cls) == 0 and np.sum(gt_cls) > 0:
            dsc = 0.0  # 如果GT有该类但预测没有，DSC定义为0
            dsc_list.append(dsc)
            continue
        try:
            dsc = dc(mask_cls, gt_cls)
        except Exception:
            dsc = np.nan
        dsc_list.append(dsc)
    return dsc_list

def sen_coefficient(mask, gt, num_classes):
    """
    使用medpy库计算每个类别的sen
    :param mask: 预测mask，二维数组
    :param gt: GT mask，二维数组
    :param num_classes: 类别数
    :return: 每个类别的sen列表
    """
    sen_list = []
    for cls in range(num_classes):
        mask_cls = (mask == cls)
        gt_cls = (gt == cls)
        # medpy的dc函数要求输入为bool类型
        if np.sum(gt_cls) == 0 and np.sum(mask_cls) == 0:
            sen = np.nan  # 如果GT和预测都没有该类，sen定义为1
            sen_list.append(sen)
            continue
        if np.sum(mask_cls) == 0 and np.sum(gt_cls) > 0:
            sen = 0.0  # 如果GT有该类但预测没有，sen定义为0
            sen_list.append(sen)
            continue
        try:
            sen = sensitivity(mask_cls, gt_cls)
        except Exception:
            sen = np.nan
        sen_list.append(sen)
    return sen_list

def hd95_coefficient(mask, gt, num_classes):
    """
    使用medpy库计算每个类别的HD95
    :param mask: 预测mask，二维数组（类别编号）
    :param gt: GT mask，二维数组（类别编号）
    :param num_classes: 类别数
    :return: 每个类别的HD95列表
    """
    hd_list = []
    for cls in range(num_classes):
        mask_cls = (mask == cls)
        gt_cls = (gt == cls)
        if np.sum(gt_cls) == 0 and np.sum(mask_cls) == 0:
            hd = np.nan  # 双方都为空，设为NaN
            hd_list.append(hd)
            continue
        try:
            hd = hd95(mask_cls, gt_cls)
        except Exception:
            hd = np.nan
        hd_list.append(hd)
    return hd_list


def compute_region_metrics(mask, gt):
    """Compute Dice/Sensitivity/HD95 for ET, TC, WT regions on one sample."""
    region_pairs = [
        (mask == 3, gt == 3),
        (np.isin(mask, [2, 3]), np.isin(gt, [2, 3])),
        (np.isin(mask, [1, 2, 3]), np.isin(gt, [1, 2, 3])),
    ]

    dscs, sens, hds = [], [], []
    for mask_region, gt_region in region_pairs:
        if np.sum(gt_region) == 0 and np.sum(mask_region) == 0:
            dscs.append(np.nan)
            sens.append(np.nan)
            hds.append(np.nan)
            continue

        if np.sum(mask_region) == 0 and np.sum(gt_region) > 0:
            dscs.append(0.0)
            sens.append(0.0)
        elif np.sum(gt_region) == 0:
            dscs.append(0.0)
            sens.append(np.nan)
        else:
            try:
                dscs.append(dc(mask_region, gt_region))
            except Exception:
                dscs.append(np.nan)
            try:
                sens.append(sensitivity(mask_region, gt_region))
            except Exception:
                sens.append(np.nan)

        try:
            hds.append(hd95(mask_region, gt_region))
        except Exception:
            hds.append(np.nan)

    return dscs, sens, hds

def process_single(mask_path, gt_dir, num_classes):
    """Worker function to process one pair of files and compute metrics.
    Returns (dscs, sens, hds) or None on failure."""
    try:
        basename = os.path.basename(mask_path)
        gt_path = os.path.join(gt_dir, basename)
        with Image.open(mask_path) as mask_file:
            mask_img = np.asarray(mask_file)
        with Image.open(gt_path) as gt_file:
            gt_img = np.asarray(gt_file)
        if mask_img.size == 0 or gt_img.size == 0:
            raise ValueError(f"Could not read prediction/ground-truth pair: {basename}")
        mask = mask_img if mask_img.ndim == 2 else mask_img[:, :, 0]
        gt = gt_img if gt_img.ndim == 2 else gt_img[:, :, 0]
        dscs, sens, hds = compute_region_metrics(mask, gt)
        return (dscs, sens, hds)
    except Exception as exc:
        raise RuntimeError(f"Failed to evaluate {mask_path}") from exc


def eval_folder_metrics(mask_dir, gt_dir, num_classes=4, num_workers=None):
    """
    计算mask文件夹和GT文件夹之间的DSC均值和标准差（使用多进程）
    :param mask_dir: 预测mask文件夹路径
    :param gt_dir: GT文件夹路径
    :param num_classes: 类别数
    :param num_workers: 并行进程数（默认：cpu_count()-1）
    :return: 每个类别的DSC均值和标准差，以及SEN和HD95的均值与标准差
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Prediction directory does not exist: {mask_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))
    mask_names = {os.path.basename(path) for path in mask_files}
    if not mask_files:
        raise ValueError(f"No PNG predictions found in: {mask_dir}")
    missing_ground_truth = sorted(name for name in mask_names if not os.path.isfile(os.path.join(gt_dir, name)))
    if missing_ground_truth:
        raise ValueError(f"Predictions without ground truth: {missing_ground_truth[:5]}")

    region_count = len(REGION_NAMES)
    all_dsc = [[] for _ in range(region_count)]
    all_sen = [[] for _ in range(region_count)]
    all_hd95 = [[] for _ in range(region_count)]

    if num_workers is None:
        num_workers = max(1, cpu_count()-1)

    worker = partial(process_single, gt_dir=gt_dir, num_classes=num_classes)
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(worker, mask_files), total=len(mask_files), ncols=90):
            if res is None:
                continue
            dscs, sens, hds = res
            for i in range(region_count):
                if not np.isnan(dscs[i]):
                    all_dsc[i].append(dscs[i])
                if not np.isnan(sens[i]):
                    all_sen[i].append(sens[i])
                if not np.isnan(hds[i]):
                    all_hd95[i].append(hds[i])

    mean_dsc = [float(np.mean(dscs)) if dscs else np.nan for dscs in all_dsc]
    std_dsc = [float(np.std(dscs)) if dscs else np.nan for dscs in all_dsc]
    mean_sen = [float(np.mean(sens)) if sens else np.nan for sens in all_sen]
    std_sen = [float(np.std(sens)) if sens else np.nan for sens in all_sen]
    mean_hd95 = [float(np.mean(hd)) if hd else np.nan for hd in all_hd95]
    std_hd95 = [float(np.std(hd)) if hd else np.nan for hd in all_hd95]
    return mean_dsc, std_dsc, mean_sen, std_sen, mean_hd95, std_hd95


def format_float_value(value, precision=2):
    if np.isnan(value):
        return "nan"
    q = Decimal("1." + "0" * precision)
    return str(Decimal(str(float(value))).quantize(q, rounding=ROUND_HALF_UP))


def format_mean_std(mean_value, std_value, is_percent=False):
    if np.isnan(mean_value) or np.isnan(std_value):
        return "nan"
    if is_percent:
        mean_text = format_float_value(mean_value * 100.0, precision=2)
        std_text = format_float_value(std_value * 100.0, precision=2)
        return f"{mean_text}%±{std_text}%"
    return f"{format_float_value(mean_value, precision=2)}±{format_float_value(std_value, precision=2)}"


def safe_nanmean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


parser = argparse.ArgumentParser(description="计算mask文件夹和GT文件夹的DSC均值和标准差")
parser.add_argument('--mask_folder', type=str, required=True, help='prediction-mask directory')
parser.add_argument('--gt_folder', type=str, required=True, help='ground-truth-mask directory')
parser.add_argument('--num_classes', type=int, default=4, help='类别数')
parser.add_argument('--num_workers', type=int, default=24, help='并行进程数（默认：cpu_count()-1）')
parser.add_argument('--csv_path', type=str, required=True, help='output CSV path')
args = parser.parse_args()

if __name__ == "__main__":
    mask_folder = args.mask_folder
    gt_folder = args.gt_folder
    csv_path = args.csv_path
    # gt_folder = "SanyuanRuanguData/knee_sag/SegmentationClassAug"
    num_classes = args.num_classes  # 根据实际类别数调整
    mean_dsc, std_dsc, mean_sen, std_sen, mean_hd95, std_hd95 = eval_folder_metrics(mask_folder, gt_folder, num_classes)

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ "【GT文件夹】", gt_folder,"【Mask文件夹】", mask_folder])
        writer.writerow(["Region", "Dice±std", "HD95±std", "Sensitivity±std"])
        for i, region_name in enumerate(REGION_NAMES):
            writer.writerow([
                region_name,
                format_mean_std(mean_dsc[i], std_dsc[i], is_percent=True),
                format_mean_std(mean_hd95[i], std_hd95[i], is_percent=False),
                format_mean_std(mean_sen[i], std_sen[i], is_percent=True),
            ])
        writer.writerow([
            "Average",
            format_mean_std(safe_nanmean(mean_dsc), safe_nanmean(std_dsc), is_percent=True),
            format_mean_std(safe_nanmean(mean_hd95), safe_nanmean(std_hd95), is_percent=False),
            format_mean_std(safe_nanmean(mean_sen), safe_nanmean(std_sen), is_percent=True),
        ])
        writer.writerow([])
