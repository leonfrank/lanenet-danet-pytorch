
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


def discriminative_loss_single(prediction, correct_label, feature_dim, delta_v, delta_d, param_var, param_dist,param_reg):
    correct_label = correct_label.view(1,-1)
    reshaped_pred = prediction.view(feature_dim, -1)
    #print("correct_label",correct_label.size(),"pred",reshaped_pred.size())
    unique_labels,unique_ids = torch.unique(correct_label,sorted=True,return_inverse=True)
    if torch.cuda.is_available():
        unique_labels = unique_labels.cuda().type(torch.cuda.LongTensor)
        unique_ids = unique_ids.cuda().type(torch.cuda.LongTensor)
    num_instances = unique_labels.size()[0]
    #print("num_instances",num_instances)
    segment_mean = torch.zeros((feature_dim,num_instances),dtype=torch.float32)
    if torch.cuda.is_available():
        segment_mean = segment_mean.cuda()
    for i,lb in enumerate(unique_labels):
        mask = correct_label.eq(lb).repeat(feature_dim,1)
        segment_embedding = torch.masked_select(reshaped_pred,mask).view(feature_dim,-1)
        segment_mean[:,i] = torch.mean(segment_embedding,dim=1)
        #print("mask",i,mask.sum())
    #print("unique instances",unique_ids.size())

    unique_ids = unique_ids.view(-1)
    #print(segment_mean)
    mu_expand = segment_mean.index_select(1,unique_ids)
    #print("mu_expand", mu_expand.size())
    distance = mu_expand-reshaped_pred
    distance = distance.norm(2,0,keepdim=True)
    distance = distance - delta_v
    distance = F.relu(distance)
    distance = distance**2
#     print("distance", distance.size())
    #组内距离
    l_var = torch.empty(num_instances,dtype=torch.float32)
    if torch.cuda.is_available():
        l_var = l_var.cuda()
    for i,lb in enumerate(unique_labels):
        mask = correct_label.eq(lb)
#         print("mask", mask.size())
        var_sum = torch.masked_select(distance,mask)
        l_var[i] = torch.mean(var_sum)
    l_var = torch.mean(l_var)
#     print(segment_mean[0])
#segment_mean_shape:[feature_dim,num_instances]
    seg_interleave = segment_mean.permute(1,0).repeat(num_instances,1)
    seg_band = segment_mean.permute(1,0).repeat(1,num_instances).view(-1,feature_dim)
    #组间距离
    dist_diff = seg_interleave - seg_band
    mask = (1-torch.eye(num_instances,dtype = torch.int8)).view(-1,1).repeat(1,feature_dim)
    if torch.cuda.is_available():
        mask = mask.cuda().type(torch.cuda.ByteTensor)
    dist_diff = torch.masked_select(dist_diff,mask).view(-1,feature_dim)
    dist_norm = dist_diff.norm(2,1)
    dist_norm = 2*delta_d - dist_norm
    dist_norm = F.relu(dist_norm)
    dist_norm = dist_norm**2
    l_dist = torch.mean(dist_norm)
    #正则化项
    l_reg = torch.mean(torch.norm(segment_mean,2,0))

    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg*l_reg
    loss = l_var + l_dist + l_reg

    return loss, l_var, l_dist, l_reg


# In[ ]:


def discriminative_loss(prediction, correct_label, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg):
    loss_batch = torch.zeros(prediction.size()[0],dtype = torch.float32)
    l_var_batch= torch.zeros(prediction.size()[0],dtype = torch.float32)
    l_dist_batch = torch.zeros(prediction.size()[0],dtype = torch.float32)
    l_reg_batch = torch.zeros(prediction.size()[0],dtype = torch.float32)
    if torch.cuda.is_available():
        loss_batch = loss_batch.cuda()
        l_var_batch = l_var_batch.cuda()
        l_dist_batch = l_dist_batch.cuda()
        l_reg_batch = l_reg_batch.cuda()
    for i in range(prediction.size()[0]):
        loss_batch[i],l_var_batch[i],l_dist_batch[i],l_reg_batch[i] = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,                                delta_v, delta_d, param_var, param_dist,param_reg)
    loss_batch = torch.mean(loss_batch)
    l_var_batch = torch.mean(l_var_batch)
    l_dist_batch = torch.mean(l_dist_batch)
    l_reg_batch = torch.mean(l_reg_batch)
#    print("l_var",l_var_batch)
#    print("l_dist",l_dist_batch)
#    print("l_reg",l_reg_batch)
    return loss_batch, l_var_batch, l_dist_batch, l_reg_batch



