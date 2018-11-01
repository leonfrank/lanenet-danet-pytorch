
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init

# In[4]:


class VGGEncoder(nn.Module):
    def __init__(self,num_blocks,in_channels, out_channels):
        super(VGGEncoder,self).__init__()
        self.num_blocks = num_blocks
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._conv_reps = [2,2,3,3,3]        
        self.net = nn.Sequential()
        for i in range(num_blocks):
            self.net.add_module("block"+str(i+1),self._encode_block(i+1))

    def _encode_block(self,block_id,kernel_size=3, stride=1):        
        out_channels = self._out_channels[block_id-1]
        padding = (kernel_size-1)//2
        seq = nn.Sequential()
        
        for i in range(self._conv_reps[block_id-1]):
            if i== 0:
                in_channels = self._in_channels[block_id-1]
            else:
                in_channels = out_channels
            seq.add_module("conv_{}_{}".format(block_id,i+1),\
                            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding))
            seq.add_module("bn_{}_{}".format(block_id,i+1),nn.BatchNorm2d(out_channels))
            seq.add_module("relu_{}_{}".format(block_id,i+1),nn.ReLU())
        seq.add_module("maxpool"+str(block_id),nn.MaxPool2d(kernel_size=2,stride=2))
        return seq

    def forward(self,input_tensor):
        ret = OrderedDict()
        #5 stage of encoding
        X = input_tensor
        for i,block in enumerate(self.net):
            pool = block(X)
#             print("pool"+str(i+1),pool.size())
            ret["pool"+str(i+1)] = pool
            
            X = pool
        return ret

