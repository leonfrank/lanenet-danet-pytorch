import os
import os.path as ops
import argparse
import time
import math
import sys
#print(sys.path)
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from lanenet_model.lanenet_merge_model import *
#from utils.data_loader import LaneDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lanenet_model.discriminative_loss import *
from utils import AverageMeter
import torch.backends.cudnn as cudnn
from utils.data_loader import *
import json


def lane_cluster_single(binary_seg_pred, pix_embedding, feature_dim = 3, delta_v = 0.5, delta_d = 1.5):
    b_seg = binary_seg_pred.squeeze(0)
    pix_embedding = pix_embedding.permute(1, 2, 0)
    count = 0
    while True:
        remaining = b_seg.eq(1).nonzero()
        if (remaining.numel() == 0):
            break
        center = remaining[0]
        center_emb = pix_embedding[center[0], center[1], :]
        eps = 1e-3
        var = 1
        while var > eps:

            dist = pix_embedding - center_emb
            dist = dist.norm(2, dim = -1)

            mask = (dist <= delta_d) * (b_seg.eq(1))
        # the embedding distance of points belongs to the same cluster should be within delta_d
            mask = mask.unsqueeze(-1).repeat(1, 1, feature_dim)
            seg_mean = pix_embedding.masked_select(mask).view(-1, feature_dim).mean(dim = 0)
            var = (seg_mean - center_emb).norm(2)
            center_emb = seg_mean
        # refine cluster result
        dist = pix_embedding - seg_mean
        dist = dist.norm(2, dim = -1)

        count -= 1
        b_seg[(dist <= delta_d) * (b_seg.eq(1))] = count
    return b_seg, -1 * count

def lane_cluster(image_data, binary_seg_pred, pix_embedding, original_size,\
                          img_name, feature_dim = 3, delta_v = 0.5, delta_d = 1.5):
    batch_size = binary_seg_pred.size()[0]
    instance_pred_batch = torch.zeros((batch_size, binary_seg_pred.size()[-2], binary_seg_pred.size()[-1]))
    lane_count_batch = np.zeros(batch_size)
    pred_batch = []
    if torch.cuda.is_available():
        instance_pred_batch = instance_pred_batch.cuda()
    for i in range(batch_size):
        instance_pred_batch[i], lane_count_batch[i] = lane_cluster_single(binary_seg_pred[i], pix_embedding[i], feature_dim, delta_v, delta_d)
    for i in range(batch_size):
        lanes = {}
        lanes["image"] = img_name[i]
        result = []
        h, w = original_size[i]
        count = int(lane_count_batch[i])
        if (count > 0):
            ins_pred = (-1 * instance_pred_batch[i]).data.cpu().numpy()
            ins_pred = cv2.resize(ins_pred, (w, h), interpolation = cv2.INTER_NEAREST)
            for c in range(1, count + 1):
                seg = (ins_pred == c).astype(np.int32)
                tmp = []
                y_proj = np.sum(seg, axis = 1)
                ##找到高度方向上第一个和最后一个点的纵坐标
                first = np.where(y_proj != 0)[0][0]
                last = np.where(y_proj != 0)[0][-1]
                for r in np.arange(first, last, 10):
                    row = seg[r,:]
                    idx = np.where(row == 1)[0]
                    #对分割的车道线用水平线截取，取截得线段的中点横坐标
                    if idx.shape[0] > 0:
                        idx = idx.mean()
                        tmp.append({"y":float(r), "x":float(idx)})
                result.append(tmp)
                result = [x for x in result if len(x) > 1]
        lanes["lanes"] = result
        pred_batch.append(lanes)
    return pred_batch

def spline_output():
    global best_epoch
    global args
    test_dataset = LaneDataSet(args["data"], "test")
    test_loader = DataLoader(test_dataset, batch_size = 8 * gpu_count, \
                            num_workers = 8, pin_memory = True)
    model = LaneNet()
    if not args["pretrained"]:
        pretrained = os.path.join(args["save"],"train_{}_epochs.pkl".format(best_epoch))
    else:
        pretrained = args["pretrained"]
    saved = torch.load(pretrained)
    model.load_state_dict({k:v for k,v in saved.items() if k in model.state_dict()})
    output_path = args["output"]
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if gpu_count >= 1:
        model = nn.DataParallel(model).cuda()
    model.eval()
    pred = []
    for batch_idx,input_data in enumerate(test_loader):
        image_data = Variable(input_data["input_tensor"]).float().cuda()
        net_output = model(image_data)
        pred_batch = lane_cluster(image_data, net_output["binary_seg_pred"], net_output["instance_seg_logits"], \
                              input_data["o_size"], input_data["img_name"])
        pred.extend(pred_batch)
    with open(os.path.join(output_path, "pred.json"),"w") as fout:
        json.dump(pred, fout, indent = 4)

gpu_count = 8
best_iou = 0
best_epoch = 25
args = {}
args["save"] = "../best/model/"
args["output"] = "../best/json/"
args["data"] = "/data/mc_data/AIC/"
args['pretrained'] = "../best/model/lanenet_aic_bestmodel.pkl"
if __name__ == '__main__':
    #main()
    spline_output()
