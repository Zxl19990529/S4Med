import os
import sys
import time
import shutil
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.utils import data
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
import socket
import time
import scipy.io as io
import cv2,argparse
from utils import get_model,get_dataloader
import basic_function as func
import dataset
import transform_exten


# os.environ['CUDA_VISIBLE_DEVICES']='7'

parser = argparse.ArgumentParser(description='PyTorch Hierachy_dif Training')
parser.add_argument('--layers', type=int, metavar='LAYERS',default=101, choices=[50, 101, 152], help='ResNet depth for deeplabv3p')
parser.add_argument('--dataset_path', metavar='DATASET_PATH',default='data/ScribbleBraTS', help='path to the ScribbleBraTS dataset')
parser.add_argument('--dataset_type', metavar='DATASET',default='BraTS', help='dataset: BraTS|MyoPS')
parser.add_argument('--numclasses', type=int, metavar='NUMCLASSES', default=4, help='number of classes')
parser.add_argument('--modality', nargs='+', type=int, default=[0,1,2,3])
parser.add_argument('--model_type',default='deeplabv3p',help='Model type selection. deeplabv3p_lorm|deeplabv3p|deeplabv2|deeplabv2_r38')
parser.add_argument('--workers', default=4, type=int, metavar='WORKERS', help='number of dataload worker')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--batchsize', default=7, type=int, metavar='BATCH_SIZE', help='batchsize')
# val param
parser.add_argument('--checkpoint_path', required=True, metavar='CHECKPOINT_PATH', help='path to the checkpoint file')
parser.add_argument('--save_path', default='./outputs/eval_ScribbleBraTS', metavar='SAVE_PATH', help='path to save predictions')
parser.add_argument('--colormap', default='BraTS', help='Choose the color map  for visualization,knee_sag')

# parser.add_argument('--crf',default=False,type=bool,help='whether use crf for post processing.')

args = parser.parse_args()

print(args)
def ss_test(model, inputs, scale=1):
    # print("tensor...")
    b, c, h, w = inputs.shape
    scaled_inputs = F.interpolate(inputs, size=(int(h*scale), int(w*scale)), mode="bilinear", align_corners=True)
    # print(inputs.shape,scaled_inputs.shape,scale)
    outputs = model.forward_eval(scaled_inputs)
    # torch.cuda.synchronize()
    # outputs = outputs[-1]
    outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
    return outputs

def ms_test(model, img):
    n, c, h, w = img.size()
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # scales = [2.0]
    inputs=img
    # full_probs = torch.cuda.FloatTensor(n, 81, h, w).fill_(0)
    # full_probs = torch.zeros((n,81,h,w),dtype=torch.FloatTensor,device='cuda')
    full_probs = None
    for scale in scales:
        # print(scale)
        probs = ss_test(model,inputs, scale)
        flip_probs = ss_test(model,inputs.flip(3), scale)
        probs = probs + flip_probs.flip(3)
        full_probs = probs if full_probs == None else (full_probs+probs)
        # full_probs += probs
    return full_probs


def test(args, test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales): # base_size=512, crop_ are 465
    tbar = tqdm ( test_loader ,ncols=90,total=len(test_loader))
    confusion_meter = tnt.meter.ConfusionMeter( args.numclasses, normalized=False )
    confusion_meter.conf = np.zeros((args.numclasses, args.numclasses), dtype=np.int64) # I add the int64 here, COCO is too big, int32 will result in the -2,147,483,648
    model.eval()

    for i, (input_imgs, gts, img_paths) in enumerate(tbar): # input_imgs is not resized to 465,465
        input_imgs = input_imgs.to(next(model.parameters()).device, non_blocking=True)
        input_imgs = F.interpolate(input_imgs, size=(base_size), mode="bilinear", align_corners=True)
        # b,ori_h,ori_w = gt.shape
        #####---------new_eval------#
        with torch.no_grad():
            predictions = ms_test(model,input_imgs) # 1,21,h,w

        #---crop and resize back the predictions---#
        for k in range(len(gts)):
            img_path = img_paths[k]
            gt = gts[k].unsqueeze(0)
            b,ori_h,ori_w = gt.shape # 1,h,w
            prediction = predictions[k].unsqueeze(0)
            prediction = torch.argmax(prediction,dim=1,keepdim=True) # 1,21,512,512 -> 1,1,512,512
            prediction = F.interpolate(prediction.float(),(ori_h,ori_w),mode='nearest')
            prediction = prediction[0].long() # 1,1,512,512 -> 1,512,512
            valid_pixel = gt.ne(255)
            confusion_meter.add(prediction[valid_pixel], gt[valid_pixel])

            mask = func.get_mask_pallete ( prediction[0].cpu ().numpy (), 'knee_sag_forlabel' )
            img_name = os.path.basename ( img_path ) # get the image name from the path
            folder_name = img_path.split('/')[-2]
            save_folder = os.path.join(args.save_path,folder_name)
            os.makedirs(save_folder,exist_ok=True)
            save_path = os.path.join(save_folder,img_name.replace('.jpg','.png'))
            mask.save ( save_path )

            mask = func.get_mask_pallete ( prediction[0].cpu ().numpy (), args.colormap )
            img_name = os.path.basename ( img_path ) # get the image name from the path
            folder_name = img_path.split('/')[-2]
            save_folder = os.path.join(args.save_path+'_color',folder_name)
            os.makedirs(save_folder,exist_ok=True)
            save_path = os.path.join(save_folder,img_name.replace('.jpg','.png'))
            mask.save ( save_path )

            # for debug the background is -1
            confusion_matrix = confusion_meter.value()
            inter = np.diag(confusion_matrix)
            union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter


    confusion_matrix = confusion_meter.value()
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

    mean_iou_ind = inter/union
    mean_iou_all = mean_iou_ind.mean()
    mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())

    print(' * IOU_All {iou}'.format(iou=mean_iou_all))
    print(' * IOU_Ind [{}]'.format(', '.join([str(x) for x in mean_iou_ind])))
    print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))
    with open(args.save_path+'.txt','a') as f:
        f.writelines(' * IOU_All {iou}\n'.format(iou=mean_iou_all))
        f.writelines(' * IOU_Ind [{}]\n'.format(', '.join([str(x) for x in mean_iou_ind])))
        f.writelines(' * ACC_Pix {acc}\n'.format(acc=mean_acc_pix))
        f.close()

    # 计算Dice指标
    dice_ind = (2 * inter) / (2 * inter + confusion_matrix.sum(1) + confusion_matrix.sum(0) - 2 * inter + 1e-12)
    mean_dice = dice_ind.mean()
    std_dice = np.std(dice_ind)

    print(' * Dice_All {dice}'.format(dice=mean_dice))
    print(' * Dice_Ind [{}]'.format(', '.join([str(x) for x in dice_ind])))
    with open(args.save_path+'.txt','a') as f:
        f.writelines(' * Dice_All {dice}\n'.format(dice=mean_dice))
        f.writelines(' * Dice_Ind [{}]\n'.format(', '.join([str(x) for x in dice_ind])))

    # 计算召回率（Recall）
    recall_ind = inter / (confusion_matrix.sum(1).clip(min=1e-12))
    mean_recall = recall_ind.mean()

    print(' * Recall_All {recall}'.format(recall=mean_recall))
    print(' * Recall_Ind [{}]'.format(', '.join([str(x) for x in recall_ind])))
    with open(args.save_path+'.txt','a') as f:
        f.writelines(' * Recall_All {recall}\n'.format(recall=mean_recall))
        f.writelines(' * Recall_Ind [{}]\n'.format(', '.join([str(x) for x in recall_ind])))

def collate_fn(list_items):
    img = []
    label = []
    imgpath=[]
    for img_, label_, imgpath_ in list_items:
    #  print(f'x_={x_}, y_={y_}')
        img.append(img_)
        label.append(label_)
        imgpath.append(imgpath_)
    img=torch.stack(img)
    return img,label,imgpath

if __name__ == '__main__':

    opt_manualSeed = 3407
    print("Random Seed: ", opt_manualSeed)
    np.random.seed ( opt_manualSeed )
    random.seed ( opt_manualSeed )
    torch.manual_seed ( opt_manualSeed )
    torch.cuda.manual_seed_all ( opt_manualSeed )

    # cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    device = torch.device(args.device)
    # if args.crf:
    #     args.save_path +='_crf'
    if not os.path.exists ( args.save_path ) :
        os.makedirs ( args.save_path )

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    val_transform = transform_exten.Compose ( [transform_exten.ResizeImage((512,512)), transform_exten.ToTensor (),transform_exten.Normalize(mean=mean,std=std)])
    val_dataset = dataset.ScribbleBraTsDataset(mode='val', dataset_folder=args.dataset_path, modality=args.modality,transform=val_transform)
    val_loader = data.DataLoader ( val_dataset, num_workers=args.workers,
                                batch_size=args.batchsize, shuffle=False, pin_memory=True,collate_fn=collate_fn)
    model_type = args.model_type
    # if model_type == 'deeplabv3p':
    #     model = deeplabV3p.Res_Deeplab(args.numclasses,args.layers)
    model_pretrain = torch.load ( args.checkpoint_path,weights_only=False ,map_location='cpu')
    print('Modality:',args.modality,len(args.modality))
    model = get_model(args.model_type,False,args,0,device,model_pretrain,input_channels=len(args.modality))
    # if model_pretrain.get('compile'):
    #     model = torch.compile(model)
    # print(model_pretrain['state_dict'].keys())
    # model = func.param_restore_all ( model, model_pretrain['state_dict'] )
    # model.load_state_dict(model_pretrain['state_dict'])
    # model = model.cuda()

    # test(args, val_loader, model, args.numclasses, mean, std, 512, 465, 465, [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    test(args, val_loader, model, args.numclasses, mean, std, 512, 465, 465, [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

    for item in args.__dict__.items():
        print(item)
