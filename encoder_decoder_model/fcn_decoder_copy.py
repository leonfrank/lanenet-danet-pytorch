
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
# In[ ]:


class FCNDecoder(nn.Module):
    
    def __init__(self,decode_layers,decode_channels,cout = 64):
        super(FCNDecoder,self).__init__()
        self._in_channels = decode_channels
        self._out_channel = 64
        self._decode_layers = decode_layers
        self.score_net = nn.Sequential()        
        self.deconv_net = nn.Sequential()        
        self.bn_net = nn.Sequential()
        for i,cin in enumerate(self._in_channels): 
            self.score_net.add_module("conv"+str(i+1), nn.Conv2d(cin,cout,1,stride=1,bias=False))
            if i > 0:
                self.deconv_net.add_module("deconv"+str(i),nn.ConvTranspose2d(cout,cout,4,stride=2,padding = 1,bias=False))
        self.deconv_last = nn.ConvTranspose2d(cout,cout,16,stride=8,padding=4,bias=False)
        self.score_last = nn.Conv2d(cout,1,1,bias=False)

    def _conv_stage(self,cin,cout):
        return nn.Sequential(nn.Conv2d(cin,cout,1,stride=1,bias=False),
                             nn.BatchNorm2d(cout),nn.ReLU())
    def _deconv_stage(self,cout):
        return nn.Sequential(nn.ConvTranspose2d(cout,cout,4,stride=2,padding = 1,bias=False),
                             nn.BatchNorm2d(cout),nn.ReLU())
    def forward(self,encode_data):
        ret = {}
        for i,layer in enumerate(self._decode_layers):
            #print(layer,encode_data[layer].size())
            if i > 0:
                deconv = self.deconv_net[i-1](score)
                #print("deconv from"+self._decode_layers[i-1],deconv.size())
            input_tensor = encode_data[layer]            
            score = self.score_net[i](input_tensor)
            #print("conv from"+layer,score.size())
            if i > 0:
                score = deconv + score
        deconv_final = self.deconv_last(score)
        score_final = self.score_last(deconv_final)
        ret['logits'] = score_final
        ret['deconv'] = deconv_final
        return ret

