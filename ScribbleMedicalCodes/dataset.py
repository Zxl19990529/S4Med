import numpy as np
import cv2
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode

from sklearn.model_selection import KFold
import torchvision.transforms.functional as F_transforms
import torchvision.transforms as transforms
import math
import time
from tqdm import tqdm

img_transforms = transforms.Compose(
    [
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ]
)
label_transforms = transforms.Compose(
    [
        transforms.CenterCrop(192)
    ]
)

class ScribbleBraTsDataset(Dataset):
    ''' modality: 0-> t1  1->t1ce  2->t2  3->flair

    '''
    def __init__(self, dataset_folder='dataset/ScribbleBraTS', fold=0, modality=[0,1, 2, 3], dstw=160, dsth=192,mode = 'train',transform=None):
        self.root_folder = dataset_folder
        self.img_dir = os.path.join(dataset_folder,'images')
        self.gt_dir = os.path.join(dataset_folder,'GT')
        self.scribble_r1 = os.path.join(dataset_folder,'scribble_r1')
        self.scribble_r2 = os.path.join(dataset_folder,'scribble_r2')
        self.scribble_r3 = os.path.join(dataset_folder,'scribble_r3')
        self.scribble_r4 = os.path.join(dataset_folder,'scribble_r4')
        self.fold = fold
        self.modality = modality
        self.modality_names = ['t1','t1ce','t2','flair']
        self.train_list = open('{}/ImageSets/SegmentationAug/{}'.format(dataset_folder, 'train.txt'),'r').read().splitlines()
        self.val_list = open('{}/ImageSets/SegmentationAug/{}'.format(dataset_folder, 'val.txt'),'r').read().splitlines()
        self.mode = mode # 'train' or 'val'
        self.transform = transform

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.val_list)

    def find_selected_names(self):
        self.selected_names = self.modality_names[self.modality]

    def get_flist(self):
        gt_list = os.listdir(self.gt_dir)
        basename_list = [filename.split('_seg.png')[0] for filename in gt_list]
        return basename_list

    def normalize(self, selected_data):
        r''' min-max normalization
        input: 4,w,h  or 1,w,h
        '''
        minmaxed_data = torch.zeros_like(selected_data, dtype=torch.float32)
        for i in range(selected_data.shape[0]):
            slice_scored = self.minmax(selected_data[i])
            minmaxed_data[i] = slice_scored
        return minmaxed_data

    def __getitem__(self,index):
        if self.mode == 'train':
            basename = self.train_list[index]
        else:
            basename = self.val_list[index]
        t1_path = os.path.join(self.img_dir,basename+'_t1.png')
        t1ce_path = os.path.join(self.img_dir,basename+'_t1ce.png')
        t2_path = os.path.join(self.img_dir,basename+'_t2.png')
        flair_path = os.path.join(self.img_dir,basename+'_flair.png')
        gt_path = os.path.join(self.gt_dir,basename+'_seg.png')
        scribble_r1_path = os.path.join(self.scribble_r1,basename+'.png')
        scribble_r2_path = os.path.join(self.scribble_r2,basename+'.png')
        scribble_r3_path = os.path.join(self.scribble_r3,basename+'.png')
        scribble_r4_path = os.path.join(self.scribble_r4,basename+'.png')

        t1_img = cv2.imread(t1_path)[:,:,[0]] #  h,w,c -> h,w,1
        t1ce_img = cv2.imread(t1ce_path)[:,:,[0]]
        t2_img = cv2.imread(t2_path)[:,:,[0]]
        flair_img = cv2.imread(flair_path)[:,:,[0]]
        gt_img = cv2.imread(gt_path)[:,:,0]
        if os.path.isfile(scribble_r1_path):
            scribble_r1_label = cv2.imread(scribble_r1_path)[:,:,0]
        else:
            scribble_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r2_path):
            scribble_r2_label = cv2.imread(scribble_r2_path)[:,:,0]
        else:
            scribble_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r3_path):
            scribble_r3_label = cv2.imread(scribble_r3_path)[:,:,0]
        else:
            scribble_r3_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r4_path):
            scribble_r4_label = cv2.imread(scribble_r4_path)[:,:,0]
        else:
            scribble_r4_label = np.ones_like(gt_img)*255

        image = np.concatenate((t1_img,t1ce_img,t2_img,flair_img),axis=2) # h,w,4
        selected_imgs = image[:,:,self.modality] # h,w,c


        if self.transform is not None:
            image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label = self.transform(selected_imgs, gt_img, scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label )

        if self.mode == 'train':
            return image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label,gt_path
        elif self.mode=='val':
            return image, label,gt_path
class ScribbleBraTS_ScribblePseudo(ScribbleBraTsDataset):
    ''' modality: 0-> C0  1->DE  2->T2

    '''
    def __init__(self, dataset_folder='dataset/ScribbleBraTS2023', fold=0, modality=[0,1, 2,3],mode = 'train',transform=None,pseudo_folder_1='None',pseudo_folder_2='None',pseudo_folder_3='None',pseudo_folder_4='None'):
        super().__init__(dataset_folder=dataset_folder, fold=fold, modality=modality,mode = mode,transform=transform)
        self.pseudo_dir_1 = os.path.join(self.root_folder,pseudo_folder_1)
        self.pseudo_dir_2 = os.path.join(self.root_folder,pseudo_folder_2)
        self.pseudo_dir_3 = os.path.join(self.root_folder,pseudo_folder_3)
        self.pseudo_dir_4 = os.path.join(self.root_folder,pseudo_folder_4)


    def __getitem__(self,index):
        if self.mode == 'train':
            basename = self.train_list[index]
        else:
            basename = self.val_list[index]
        t1_path = os.path.join(self.img_dir,basename+'_t1.png')
        t1ce_path = os.path.join(self.img_dir,basename+'_t1ce.png')
        t2_path = os.path.join(self.img_dir,basename+'_t2.png')
        flair_path = os.path.join(self.img_dir,basename+'_flair.png')
        gt_path = os.path.join(self.gt_dir,basename+'_seg.png')
        scribble_r1_path = os.path.join(self.scribble_r1,basename+'.png')
        scribble_r2_path = os.path.join(self.scribble_r2,basename+'.png')
        scribble_r3_path = os.path.join(self.scribble_r3,basename+'.png')
        scribble_r4_path = os.path.join(self.scribble_r4,basename+'.png')
        pseudo_r1_path = os.path.join(self.pseudo_dir_1,basename+'_t1.png')
        pseudo_r2_path = os.path.join(self.pseudo_dir_2,basename+'_t1ce.png')
        pseudo_r3_path = os.path.join(self.pseudo_dir_3,basename+'_t2.png')
        pseudo_r4_path = os.path.join(self.pseudo_dir_4,basename+'_flair.png')

        t1_img = cv2.imread(t1_path)[:,:,[0]] #  h,w,c -> h,w,1
        t1ce_img = cv2.imread(t1ce_path)[:,:,[0]]
        t2_img = cv2.imread(t2_path)[:,:,[0]]
        flair_img = cv2.imread(flair_path)[:,:,[0]]
        gt_img = cv2.imread(gt_path)[:,:,0]
        if os.path.isfile(scribble_r1_path):
            scribble_r1_label = cv2.imread(scribble_r1_path)[:,:,0]
        else:
            scribble_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r2_path):
            scribble_r2_label = cv2.imread(scribble_r2_path)[:,:,0]
        else:
            scribble_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r3_path):
            scribble_r3_label = cv2.imread(scribble_r3_path)[:,:,0]
        else:
            scribble_r3_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r4_path):
            scribble_r4_label = cv2.imread(scribble_r4_path)[:,:,0]
        else:
            scribble_r4_label = np.ones_like(gt_img)*255

        if os.path.isfile(pseudo_r1_path):
            pseudo_r1_label = cv2.imread(pseudo_r1_path)[:,:,0]
        else:
            pseudo_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(pseudo_r2_path):
            pseudo_r2_label = cv2.imread(pseudo_r2_path)[:,:,0]
        else:
            pseudo_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(pseudo_r3_path):
            pseudo_r3_label = cv2.imread(pseudo_r3_path)[:,:,0]
        else:
            pseudo_r3_label = np.ones_like(gt_img)*255
        if os.path.isfile(pseudo_r4_path):
            pseudo_r4_label = cv2.imread(pseudo_r4_path)[:,:,0]
        else:
            pseudo_r4_label = np.ones_like(gt_img)*255

        image = np.concatenate((t1_img,t1ce_img,t2_img,flair_img),axis=2) # h,w,4
        selected_imgs = image[:,:,self.modality] # h,w,c


        if self.transform is not None:
            image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label,pseudo_r4_label = self.transform(selected_imgs, gt_img, scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label,pseudo_r4_label )

        if self.mode == 'train':
            return image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r4_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label,pseudo_r4_label,gt_path
        elif self.mode=='val':
            return image, label,gt_path

class ScribbleMyoPS(Dataset):
    ''' modality: 0-> C0  1->DE  2->T2

    '''
    def __init__(self, dataset_folder='dataset/ScribbleMyoPS', fold=0, modality=[0,1, 2],mode = 'train',transform=None):
        self.root_folder = dataset_folder
        self.img_dir = os.path.join(dataset_folder,'images')
        self.gt_dir = os.path.join(dataset_folder,'GT')
        self.scribble_r1 = os.path.join(dataset_folder,'scribble_r1')
        self.scribble_r2 = os.path.join(dataset_folder,'scribble_r2')
        self.scribble_r3 = os.path.join(dataset_folder,'scribble_r3')
        self.fold = fold
        self.modality = modality
        self.modality_names = ['C0','DE','T2']
        self.train_list = open('{}/ImageSets/SegmentationAug/{}'.format(dataset_folder, 'train.txt'),'r').read().splitlines()
        self.val_list = open('{}/ImageSets/SegmentationAug/{}'.format(dataset_folder, 'val.txt'),'r').read().splitlines()
        self.mode = mode # 'train' or 'val'
        self.transform = transform
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.val_list)

    def find_selected_names(self):
        self.selected_names = self.modality_names[self.modality]

    def get_flist(self):
        gt_list = os.listdir(self.gt_dir)
        basename_list = [filename.split('_gd.png')[0] for filename in gt_list]
        return basename_list

    def __getitem__(self,index):
        if self.mode == 'train':
            basename = self.train_list[index]
        else:
            basename = self.val_list[index]
        C0_path = os.path.join(self.img_dir,basename+'_C0.png')
        t2_path = os.path.join(self.img_dir,basename+'_T2.png')
        DE_path = os.path.join(self.img_dir,basename+'_DE.png')
        gt_path = os.path.join(self.gt_dir,basename+'_gd.png')
        scribble_r1_path = os.path.join(self.scribble_r1,basename+'.png')
        scribble_r2_path = os.path.join(self.scribble_r2,basename+'.png')
        scribble_r3_path = os.path.join(self.scribble_r3,basename+'.png')

        C0_img = cv2.imread(C0_path)[:,:,[0]] #  h,w,c -> h,w,1
        t2_img = cv2.imread(t2_path)[:,:,[0]]
        DE_img = cv2.imread(DE_path)[:,:,[0]]

        gt_img = cv2.imread(gt_path)[:,:,0]
        if os.path.isfile(scribble_r1_path):
            scribble_r1_label = cv2.imread(scribble_r1_path)[:,:,0]
        else:
            scribble_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r2_path):
            scribble_r2_label = cv2.imread(scribble_r2_path)[:,:,0]
        else:
            scribble_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r3_path):
            scribble_r3_label = cv2.imread(scribble_r3_path)[:,:,0]
        else:
            scribble_r3_label = np.ones_like(gt_img)*255

        imgs = np.concatenate((C0_img,t2_img,DE_img),axis=2) # h,w,3
        selected_imgs = imgs[:,:,self.modality] # h,w,c


        if self.transform is not None:
            image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,_ = self.transform(selected_imgs, gt_img, scribble_r1_label,scribble_r2_label,scribble_r3_label,scribble_r3_label )

        if self.mode == 'train':
            return image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,_,gt_path
        else:
            return image, label,gt_path

class ScribbleMyoPS_ScribblePseudo(ScribbleMyoPS):
    ''' modality: 0-> C0  1->DE  2->T2

    '''
    def __init__(self, dataset_folder='dataset/ScribbleMyoPS', fold=0, modality=[0,1, 2],mode = 'train',transform=None,pseudo_folder_1='None',pseudo_folder_2='None',pseudo_folder_3='None'):
        super().__init__(dataset_folder=dataset_folder, fold=fold, modality=modality,mode = mode,transform=transform)
        self.pseudo_dir_1 = os.path.join(self.root_folder,pseudo_folder_1)
        self.pseudo_dir_2 = os.path.join(self.root_folder,pseudo_folder_2)
        self.pseudo_dir_3 = os.path.join(self.root_folder,pseudo_folder_3)


    def __getitem__(self,index):
        if self.mode == 'train':
            basename = self.train_list[index]
        else:
            basename = self.val_list[index]
        C0_path = os.path.join(self.img_dir,basename+'_C0.png')
        t2_path = os.path.join(self.img_dir,basename+'_T2.png')
        DE_path = os.path.join(self.img_dir,basename+'_DE.png')
        gt_path = os.path.join(self.gt_dir,basename+'_gd.png')
        scribble_r1_path = os.path.join(self.scribble_r1,basename+'.png')
        scribble_r2_path = os.path.join(self.scribble_r2,basename+'.png')
        scribble_r3_path = os.path.join(self.scribble_r3,basename+'.png')
        pseudo_r1_path = os.path.join(self.pseudo_dir_1,basename+'_C0.png')
        pseudo_r2_path = os.path.join(self.pseudo_dir_2,basename+'_DE.png')
        pseudo_r3_path = os.path.join(self.pseudo_dir_3,basename+'_T2.png')

        C0_img = cv2.imread(C0_path)[:,:,[0]] #  h,w,c -> h,w,1
        t2_img = cv2.imread(t2_path)[:,:,[0]]
        DE_img = cv2.imread(DE_path)[:,:,[0]]

        gt_img = cv2.imread(gt_path)[:,:,0]
        if os.path.isfile(scribble_r1_path):
            scribble_r1_label = cv2.imread(scribble_r1_path)[:,:,0]
        else:
            scribble_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r2_path):
            scribble_r2_label = cv2.imread(scribble_r2_path)[:,:,0]
        else:
            scribble_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(scribble_r3_path):
            scribble_r3_label = cv2.imread(scribble_r3_path)[:,:,0]
        else:
            scribble_r3_label = np.ones_like(gt_img)*255

        if os.path.isfile(pseudo_r1_path):
            pseudo_r1_label = cv2.imread(pseudo_r1_path)[:,:,0]
        else:
            pseudo_r1_label = np.ones_like(gt_img)*255
        if os.path.isfile(pseudo_r2_path):
            pseudo_r2_label = cv2.imread(pseudo_r2_path)[:,:,0]
        else:
            pseudo_r2_label = np.ones_like(gt_img)*255
        if os.path.isfile(pseudo_r3_path):
            pseudo_r3_label = cv2.imread(pseudo_r3_path)[:,:,0]
        else:
            pseudo_r3_label = np.ones_like(gt_img)*255


        imgs = np.concatenate((C0_img,t2_img,DE_img),axis=2) # h,w,3
        selected_imgs = imgs[:,:,self.modality] # h,w,c


        if self.transform is not None:
            image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label = self.transform(selected_imgs, gt_img, scribble_r1_label,scribble_r2_label,scribble_r3_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label )

        if self.mode == 'train':
            return image, label,scribble_r1_label,scribble_r2_label,scribble_r3_label,pseudo_r1_label,pseudo_r2_label,pseudo_r3_label,gt_path
        else:
            return image, label,gt_path

# if __name__ == '__main__':
#     train_dataset = ScribbleBraTsDataset(dataset_folder='dataset/ScribbleBraTS',fold=2,modality=[0,1],dstw=192,dsth=192,mode='train')
#     val_dataset = ScribbleBraTsDataset(dataset_folder='dataset/ScribbleBraTS',fold=2,modality=[0,1],dstw=192,dsth=192,mode='val')

#     train_loader = DataLoader(train_dataset,batch_size=16,drop_last=True,shuffle=True,num_workers=2)

#     for imgs_tensor, scribble_tensor,gt_tensor in tqdm(train_loader):
#         # print(imgs_tensor.shape)
#         a = 1
