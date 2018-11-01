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
from utils.data_loader import LaneDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lanenet_model.discriminative_loss import *
from utils import AverageMeter
import json
import torch.backends.cudnn as cudnn

#setting state to adjust learning rate
state = {}
state["schedule"] = [100,200]
state["lr"] = 0.1
VGG_MEAN= np.array([103.939, 116.779, 123.68])
gpu_count = 8
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in state["schedule"]:
        state["lr"] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def save_model(path,epoch,model):
    if isinstance(model,nn.DataParallel):
        save = model.module
    else:
        save = model
    torch.save(save.state_dict(),os.path.join(path,"train_{}_epochs.pkl".format(epoch+1)))

def output_loss(net_output,binary_label,instance_label):
    k_binary = 0.7
    k_instance = 0.3
    loss_fn = nn.CrossEntropyLoss().cuda()
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits,binary_label)

    #binary_loss = net_output["binary_seg_loss"].sum()
    #instance_loss = net_output["disc_loss"].sum()
    pix_embedding = net_output["instance_seg_logits"]
    instance_loss, _ , _, _ = discriminative_loss(pix_embedding,instance_label, 3, 0.5, 1.5, 1.0, 1.0, 0.001)
    binary_loss = binary_loss * k_binary
    instance_loss = instance_loss * k_instance
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]
    #pix_cls = out[binary_label]
    iou = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0)*binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        iou += TP / union
    iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, out , iou

    #print("Global configuration is as follows:")

def compose_img(image_data,out,binary_label,pix_embedding,instance_label,i):
    val_gt = (image_data[i].cpu().numpy().transpose(1,2,0) + VGG_MEAN).astype(np.uint8)
    val_pred = out[i].squeeze(0).cpu().numpy().transpose(0,1)*255
    val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0,1)*255
    val_out = np.zeros((val_pred.shape[0],val_pred.shape[1],3),dtype = np.uint8)
    val_out[:,:,0] = val_pred
    val_out[:,:,1] = val_label
    val_gt[val_out == 255] = 255
    epsilon = 1e-5
    pix_embedding = pix_embedding[i].data.cpu().numpy()
    pix_vec = pix_embedding/(np.sum(pix_embedding,axis=0,keepdims=True)+epsilon)*255
    pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1,2,0)
    ins_label = instance_label[i].data.cpu().numpy().transpose(0,1)
    ins_label = np.repeat(np.expand_dims(ins_label,-1),3,-1)
    val_img = np.concatenate((val_gt,pix_vec,ins_label),axis = 0)
    return val_img

def lane_cluster_single(binary_seg_pred, pix_embedding, feature_dim = 3, delta_v = 0.5,delta_d = 1.5):
    b_seg = binary_seg_pred.squeeze(0)
    pix_embedding = pix_embedding.permute(1,2,0)
    count = 0
    while True:
        remaining = b_seg.eq(1).nonzero()
        if (remaining.numel()==0):
            break
        cur = remaining[0]
        cur_emb = pix_embedding[cur[0],cur[1],:]
        dist = pix_embedding - cur_emb
        dist = dist.norm(2,dim=-1)

        mask = (dist <= delta_d)*(b_seg.eq(1))
        if torch.cuda.is_available():
            mask = mask.cuda()
        # the embedding distance of points belongs to the same cluster should be within delta_d
        mask = mask.unsqueeze(-1).repeat(1,1,feature_dim)
        seg_mean = pix_embedding.masked_select(mask).view(-1,feature_dim).mean(dim=0)
        # refine cluster result
        dist = pix_embedding - seg_mean
        dist = dist.norm(2,dim = -1)

        count -= 1
        b_seg[(dist<= delta_d)*(b_seg.eq(1))] = count
    return b_seg, -1*count

def lane_cluster_and_draw(image_data, binary_seg_pred, pix_embedding, original_size, val_name, json_path, feature_dim = 3, delta_v = 0.5,delta_d = 1.5):
    batch_size = binary_seg_pred.size()[0]
    instance_pred_batch = torch.zeros((batch_size,binary_seg_pred.size()[-2],binary_seg_pred.size()[-1]))
    lane_count_batch = np.zeros(batch_size)
    if torch.cuda.is_available():
        instance_pred_batch = instance_pred_batch.cuda()
    for i in range(batch_size):
        instance_pred_batch[i],lane_count_batch[i] = lane_cluster_single(binary_seg_pred[i], pix_embedding[i], feature_dim, delta_v ,delta_d)
    for i in range(batch_size):
        lanes = []
        h,w = original_size[i]
        gt_image = cv2.resize((image_data[i].cpu().numpy().transpose(1,2,0) + VGG_MEAN).astype(np.uint8),(w,h))
        count = int(lane_count_batch[i])
        #print(count)
        if (count > 0):
            ins_pred = (-1*instance_pred_batch[i]).data.cpu().numpy()
            ins_pred = cv2.resize(ins_pred,(w,h),interpolation = cv2.INTER_NEAREST)
            for c in range(1,count+1):
                tmp = []
                for r in np.arange(0,h,5):#spot the point every 10 pixels in height
                    row = ins_pred[r,:]
                    idx = np.where(row == c)[0]
                    if idx.shape[0]>0:
                        idx = idx.mean()
                        tmp.append({"y":float(r), "x":float(idx)})
                lanes.append(tmp)

            for line in lanes:
                tmp = []
                for point in line:
                    tmp.append((int(point["x"]),int(point["y"])))
                tmp = np.asarray(tmp)
                rnd = np.random.randint(255,size=3)
                cv2.polylines(gt_image,[tmp],0,(int(rnd[0]),int(rnd[1]),int(rnd[2])),5)
#            cv2.imwrite("exlines/"+val_name[i]+".png",gt_image)
        with open("{}/{}.json".format(json_path,val_name[i]),"w") as f:
            json.dump({"Lanes":lanes},f,indent=4)

def train(train_loader,model,optimizer,im_path, epoch):
    model.train()
    step = 0
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    for batch_idx,input_data in enumerate(train_loader):
        step += 1

        image_data = Variable(input_data["input_tensor"]).cuda().type(torch.cuda.FloatTensor)
        instance_label = Variable(input_data["instance_label"]).cuda().type(torch.cuda.LongTensor)
        binary_label = Variable(input_data["binary_label"]).cuda().type(torch.cuda.LongTensor)

        #output process
        net_output = model(image_data)
        total_loss, binary_loss, instance_loss, out, train_iou = output_loss(net_output, binary_label, instance_label)
        total_losses.update(total_loss.item(),image_data.size()[0])
        binary_losses.update(binary_loss.item(),image_data.size()[0])
        instance_losses.update(instance_loss.item(),image_data.size()[0])
        mean_iou.update(train_iou,image_data.size()[0])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if np.isnan(total_loss.item()) or np.isnan(binary_loss.item()) or np.isnan(instance_loss.item()):
            print('cost is: {:.5f}'.format(total_loss.item()))
            print('binary cost is: {:.5f}'.format(binary_loss.item()))
            print('instance cost is: {:.5f}'.format(instance_loss.item()))
            cv2.imwrite('nan_image.png', image_data[0].cpu().numpy().transpose(1,2,0) + VGG_MEAN)
            cv2.imwrite('nan_instance_label.png', image_data[0].cpu().numpy().transpose(1,2,0))
            cv2.imwrite('nan_binary_label.png', binary_label[0].cpu().numpy().transpose(1,2,0) * 255)
            cv2.imwrite('nan_embedding.png', pix_embedding[0].cpu().numpy().transpose(1,2,0))
            break
        if step%500 == 0:
            print("Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
            ep=epoch+1,
            st = step,
            batch = batch_idx+1,
            size = len(train_loader),
            et = batch_time.val,
            tot = total_losses.avg,
            bin = binary_losses.avg,
            ins = instance_losses.avg,
            iou = train_iou,
            ))
            sys.stdout.flush()
            train_img_list = []
            for i in range(3):
                train_img_list.append(compose_img(image_data,out,binary_label,net_output["instance_seg_logits"],instance_label,i))
            train_img = np.concatenate(train_img_list,axis=1)
            cv2.imwrite(os.path.join(im_path,"train_"+str(epoch+1)+"_step_"+str(step)+".png"),train_img)
    return mean_iou.avg

def test(val_loader,model,im_path, json_path, epoch):
    model.eval()
    step = 0
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    mean_iou = AverageMeter()
    end = time.time()
    val_img_list = []
    val_img_md5 = open(os.path.join(im_path,"val_"+str(epoch+1)+".txt"),"w")
    for batch_idx,input_data in enumerate(val_loader):
        step += 1
        image_data = Variable(input_data["input_tensor"]).cuda().type(torch.cuda.FloatTensor)
        instance_label = Variable(input_data["instance_label"]).cuda().type(torch.cuda.LongTensor)
        binary_label = Variable(input_data["binary_label"]).cuda().type(torch.cuda.LongTensor)

        #output process
        net_output = model(image_data)
        total_loss, binary_loss, instance_loss, out, val_iou = output_loss(net_output, binary_label, instance_label)
        total_losses.update(total_loss.item(),image_data.size()[0])
        binary_losses.update(binary_loss.item(),image_data.size()[0])
        instance_losses.update(instance_loss.item(),image_data.size()[0])
        mean_iou.update(val_iou,image_data.size()[0])

        if step%100 == 0:
            val_img_list.append(compose_img(image_data,out,binary_label,net_output["instance_seg_logits"],instance_label,0))
            val_img_md5.write(input_data["img_name"][0]+"\n")
#        lane_cluster_and_draw(image_data, net_output["binary_seg_pred"], net_output["instance_seg_logits"], input_data["o_size"], input_data["img_name"], json_path)
    batch_time.update(time.time() - end)
    end = time.time()

    print("Epoch {ep} Validation Report | ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
    ep=epoch+1,
    et = batch_time.val,
    tot = total_losses.avg,
    bin = binary_losses.avg,
    ins = instance_losses.avg,
    iou = mean_iou.avg,
    ))
    sys.stdout.flush()
    val_img = np.concatenate(val_img_list,axis=1)
    cv2.imwrite(os.path.join(im_path,"val_"+str(epoch+1)+".png"),val_img)
    val_img_md5.close()
    return mean_iou.avg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",help="Directory of the AIC dataset")
    parser.add_argument("--save",help="Directory to save model checkpoint")
    parser.add_argument("--epochs",type=int,help="Training epochs")
    parser.add_argument("--pretrained",required = False,default=None,help="pretrained model path")
    parser.add_argument("--image",help = "output image folder")
    parser.add_argument("--net",help = "backbone network")
    parser.add_argument("--json",help = "post processing json")
    return parser.parse_args()

def load_imagenet(model, pretrained = './encoder.pkl'):
    pre_weights = torch.load(pretrained)
    model_weights = model._encoder.state_dict()
    model_weights.update(pre_weights)
    model._encoder.load_state_dict(model_weights)

best_iou = 0

def main():
    global best_iou
    args = parse_args()
    start_epoch = 0
    save_path = args.save
    im_path = args.image
    json_path = args.json
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(im_path):
        os.makedirs(im_path)
    if not os.path.isdir(json_path):
        os.makedirs(json_path)
    train_dataset = LaneDataSet(args.dataset,"train")
    val_dataset = LaneDataSet(args.dataset,"val")
    model = LaneNet()
    load_imagenet(model)
    train_loader = DataLoader(train_dataset,batch_size= 8 * gpu_count,num_workers = 8,pin_memory = True,shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size= 8 * gpu_count,num_workers = 8,pin_memory = True,shuffle = True)
    if args.pretrained:
        saved = torch.load(args.pretrained)
        model.load_state_dict({k:v for k,v in saved.items() if k in model.state_dict()})
        start_epoch = int(args.pretrained.split("_")[-2])
    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # optimizer = torch.optim.SGD(model.parameters(),lr = state['lr'],momentum = 0.9,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
    for epoch in range(start_epoch,args.epochs):
        #  adjust_learning_rate(optimizer,epoch)
        train_iou = train(train_loader,model,optimizer,im_path,epoch)
        val_iou = test(val_loader,model,im_path,json_path,epoch)
        if (epoch+1)%5 == 0:
            save_model(save_path,epoch,model)
        best_iou = max(val_iou,best_iou)
        print('Best IoU : {}'.format(best_iou))

if __name__ == '__main__':
    main()
