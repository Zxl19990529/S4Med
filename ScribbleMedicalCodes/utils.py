import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from models import deeplabV3p,UNet,ResUNet,UNetplusplus
import dataset
import transform as transform
import transform_exten

def get_model(model_type,distributed,args,local_rank,device,checkpoint=None,input_channels=3):
    model = None
    if model_type == 'deeplabv3p':
        model = deeplabV3p.Res_Deeplab(args.numclasses,args.layers,input_channels)
    elif model_type == 'UNet':
        model = UNet.UNet(n_channels=input_channels,n_classes=args.numclasses)
    elif model_type == 'ResUNet':
        model = ResUNet.ResUNet(n_channels=input_channels,n_classes=args.numclasses)
    elif model_type == 'UNet++':
        model = UNetplusplus.UNetplusplus(n_channels=input_channels,n_classes=args.numclasses)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # if args.compile:
    #     model = torch.compile(model)
    if checkpoint !=None:
        if checkpoint.get('compile'):
            model = torch.compile(model)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])

    return model


def get_dataloader(args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform_exten = transform_exten.Compose([
            transform_exten.RandScale([0.5, 2.0]),
            transform_exten.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform_exten.RandomGaussianBlur(),
            transform_exten.RandomHorizontalFlip(),
            transform_exten.Crop([args.crop_size, args.crop_size], crop_type='rand', padding=mean, ignore_label=255),
            transform_exten.ToTensor(),
            transform_exten.Normalize(mean=mean, std=std)])
    train_transform_base = transform.Compose([
            transform.RandScale([0.5, 2.0]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.crop_size, args.crop_size], crop_type='rand', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_transform_base = transform.Compose([
        transform.Crop([args.crop_size, args.crop_size], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_transform_exten = transform_exten.Compose([
        transform_exten.Crop([args.crop_size, args.crop_size], crop_type='center', padding=mean, ignore_label=255),
        transform_exten.ToTensor(),
        transform_exten.Normalize(mean=mean, std=std)])
    if args.dataset_type == 'BraTS':
        if args.dataset == 'ScribbleOnly':
            train_dataset = dataset.ScribbleBraTsDataset(mode='train', dataset_folder=args.dataset_path, modality=args.modality,transform=train_transform_exten)
            val_dataset = dataset.ScribbleBraTsDataset(mode='val', dataset_folder=args.dataset_path, modality=args.modality,transform=val_transform_exten)
        elif args.dataset == 'ScribblePseudo':
            train_dataset = dataset.ScribbleBraTS_ScribblePseudo(mode='train', dataset_folder=args.dataset_path, modality=args.modality,transform=train_transform_exten,pseudo_folder_1=args.pesudo_1,pseudo_folder_2=args.pesudo_2,pseudo_folder_3=args.pesudo_3,pseudo_folder_4=args.pesudo_4)
            val_dataset = dataset.ScribbleBraTS_ScribblePseudo(mode='val', dataset_folder=args.dataset_path, modality=args.modality,transform=val_transform_exten,pseudo_folder_1=args.pesudo_1,pseudo_folder_2=args.pesudo_2,pseudo_folder_3=args.pesudo_3,pseudo_folder_4=args.pesudo_4)
    elif args.dataset_type == 'MyoPS':
        if args.dataset == 'ScribbleOnly':
            train_dataset = dataset.ScribbleMyoPS(mode='train', dataset_folder=args.dataset_path, modality=args.modality,transform=train_transform_exten)
            val_dataset = dataset.ScribbleMyoPS(mode='val', dataset_folder=args.dataset_path, modality=args.modality,transform=val_transform_exten)
        elif args.dataset == 'ScribblePseudo':
            train_dataset = dataset.ScribbleMyoPS_ScribblePseudo(mode='train', dataset_folder=args.dataset_path, modality=args.modality,transform=train_transform_exten,pseudo_folder_1=args.pesudo_1,pseudo_folder_2=args.pesudo_2,pseudo_folder_3=args.pesudo_3)
            val_dataset = dataset.ScribbleMyoPS_ScribblePseudo(mode='val', dataset_folder=args.dataset_path, modality=args.modality,transform=val_transform_exten,pseudo_folder_1=args.pesudo_1,pseudo_folder_2=args.pesudo_2,pseudo_folder_3=args.pesudo_3)
    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_train, pin_memory=True,prefetch_factor=2,drop_last=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_val, pin_memory=True,prefetch_factor=2)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True,drop_last=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=int(args.batchsize), shuffle=False, pin_memory=True)

    return train_loader,val_loader
