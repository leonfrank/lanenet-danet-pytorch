
# coding: utf-8

# In[1]:


import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import cv2
import numpy as np


# In[2]:


from torchvision.transforms import ToTensor


# In[3]:


class LaneDataSet(Dataset):
    def __init__(self,data_dir,mode):
        self.mode = mode
        self.data_dir = data_dir
        if self.mode!= "test":
            self.img_list,self.binary_label,self.instance_label = self._load_file(os.path.join(data_dir,mode+".txt"))
        else:
            self.img_list = self._load_file(os.path.join(data_dir,mode+".txt"))
    def _load_file(self,file):
        if self.mode!= "test":
            img_list = []
            binary_label = []
            instance_label = []
            with open(file,"r") as f:
                for line in f:
                    l = line.strip("\n").split()
                    img_list.append(l[0])
                    binary_label.append(l[1])
                    instance_label.append(l[2])
            return img_list,binary_label,instance_label
        else:
            img_list = []
            with open(file,"r") as f:
                for line in f:
                    l = line.strip("\n")
                    img_list.append(l)
            return img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        VGG_MEAN= np.array([103.939, 116.779, 123.68])
        # img_part_name = self.img_list[idx].split("/")[-3]+'_'+self.img_list[idx].split("/")[-2]+'_'+self.img_list[idx].split("/")[-1][:-4]
        img_part_name = self.img_list[idx].split("/")[-1][:-4]
        o_img = cv2.imread(os.path.join(self.data_dir,self.img_list[idx]))
        h,w,c = o_img.shape
        original_size = np.array([h,w])
        gt_image = cv2.resize(o_img,(512,256))
        gt_image = np.asarray(gt_image).astype(np.float32)
        gt_image -= VGG_MEAN
        gt_image = np.transpose(gt_image,(2,0,1))
        if self.mode == "test":
            return {"img_name":img_part_name,"input_tensor":gt_image,"o_size":original_size}
        gt_binary_label = cv2.imread(os.path.join(self.data_dir,self.binary_label[idx]),cv2.IMREAD_GRAYSCALE)
        gt_binary_label = gt_binary_label//255
        gt_binary_label = cv2.resize(gt_binary_label,(512,256),interpolation = cv2.INTER_NEAREST)
        gt_binary_label = torch.from_numpy(gt_binary_label)
        #print(self.instance_label[idx])
        gt_instance_label = cv2.imread(os.path.join(self.data_dir,self.instance_label[idx]),cv2.IMREAD_GRAYSCALE)
        gt_instance_label = cv2.resize(gt_instance_label,(512,256),interpolation=cv2.INTER_NEAREST)
        #print(np.unique(gt_instance_label))
        gt_instance_label = torch.from_numpy(gt_instance_label)

        sample = {"img_name":img_part_name,"o_size":original_size,"input_tensor":gt_image,"binary_label":gt_binary_label,"instance_label":gt_instance_label}

        return sample
