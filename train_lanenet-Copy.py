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
from lanenet_model.loss import *
from utils.data_loader import LaneDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lanenet_model.discriminative_loss import *

#setting state to adjust learning rate
state = {}
state["schedule"] = [500,1000]
state["lr"] = 0.01

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

def train_net(data_dir, save_path,im_path,train_epochs,pretrained_model = None,net_flag='vgg',use_gpu = True,gpu_count = 8):
    #Todo: adopt from get_train_val_loader
    train_dataset = LaneDataSet(data_dir,"train")
    val_dataset = LaneDataSet(data_dir,"val")
    model = LaneNet(net_flag)
    start_epoch = 0
    if pretrained_model:
        saved = torch.load(pretrained_model)
        model.load_state_dict({k:v for k,v in saved.items() if k in model.state_dict()})
        #start_epoch = int(pretrained_model.split("_")[-2])
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr = state['lr'],momentum=0.9,weight_decay=5e-4)
#    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    #print("Global configuration is as follows:")
    global_step = 0
    k_binary_loss = 1.
    k_instance_loss = 0.
    for epoch in range(start_epoch,train_epochs):
        t_start = time.time()
        train_loader = DataLoader(train_dataset,batch_size=4,pin_memory=True)
        adjust_learning_rate(optimizer,epoch)
        phase = "train"
        model.train()
        for input_data in train_loader:
            optimizer.zero_grad()
            global_step += 1
            image_data = Variable(input_data["input_tensor"]).cuda().type(torch.cuda.FloatTensor)
            instance_label = Variable(input_data["instance_label"]).cuda().type(torch.cuda.LongTensor)
            binary_label = Variable(input_data["binary_label"]).cuda().type(torch.cuda.LongTensor)

            #output process
            net_output = model(image_data)
            binary_seg_logits = net_output["binary_seg_logits"]
            #binary_seg_logits = torch.sigmoid(binary_seg_logits)
            #binary_loss = dice_loss_single(binary_seg_logits,binary_label)
            loss_fn = nn.CrossEntropyLoss()
            binary_loss = loss_fn(binary_seg_logits,binary_label)
            pix_embedding = net_output["instance_seg_logits"]
            instance_loss, _, _, _ = discriminative_loss(pix_embedding,instance_label, 3, 0.5, 1.5, 1.0, 1.0, 0.001)
            total_loss = binary_loss * k_binary_loss + instance_loss * k_instance_loss
            out = net_output["binary_seg_pred"]
            binary_label = binary_label.unsqueeze(1).type(torch.cuda.ByteTensor)
            pix_cls = out.masked_select(binary_label)
            train_acc = pix_cls.nonzero().size()[0]/pix_cls.size()[0]

            if np.isnan(total_loss.item()) or np.isnan(binary_loss.item()) or np.isnan(instance_loss.item()):
                print('cost is: {:.5f}'.format(total_loss.item()))
                print('binary cost is: {:.5f}'.format(binary_loss.item()))
                print('instance cost is: {:.5f}'.format(instance_loss.item()))
                cv2.imwrite('nan_image.png', image_data[0].cpu().numpy().transpose(1,2,0) + VGG_MEAN)
                cv2.imwrite('nan_instance_label.png', image_data[0].cpu().numpy().transpose(1,2,0))
                cv2.imwrite('nan_binary_label.png', binary_label[0].cpu().numpy().transpose(1,2,0) * 255)
                cv2.imwrite('nan_embedding.png', pix_embedding[0].cpu().numpy().transpose(1,2,0))
                break


            print("Epoch: {}, Elapsed Time: {:.2f}".format(epoch+1, (time.time() - t_start)/60.0),"total loss {:.5f}".format(total_loss.item()),"binary loss {:.5f}".format(binary_loss.item() * k_binary_loss),"instance loss {:.5f}".format(instance_loss.item() * k_instance_loss),"train accuracy {:.5f}".format(train_acc))
            cv2.imwrite(os.path.join(im_path,str(epoch+1)+"train_pred.png"),out[0].squeeze(0).cpu().numpy().transpose(0,1)*255)
            cv2.imwrite(os.path.join(im_path,str(epoch+1)+"train_label.png"),binary_label[0].squeeze(0).cpu().numpy().transpose(0,1)*255)

            total_loss.backward()
            optimizer.step()
            if (epoch+1)%10 == 0:
                save_model(save_path,epoch,model)
        #valid phase

        phase = "valid"

        valid_loader = DataLoader(val_dataset,batch_size=4,pin_memory=True)
        for input_data in valid_loader:
            image_data = Variable(input_data["input_tensor"]).cuda().type(torch.cuda.FloatTensor)
            instance_label = Variable(input_data["instance_label"]).cuda().type(torch.cuda.LongTensor)
            binary_label = Variable(input_data["binary_label"]).type(torch.cuda.LongTensor)
            #output process
            val_acc = []
            val_total_loss_list = []
            val_binary_loss_list = []
            val_instance_loss_list = []

            net_output = model(image_data)

            binary_seg_logits = net_output["binary_seg_logits"]
            #binary_seg_logits = torch.sigmoid(binary_seg_logits)
            #val_binary_loss = dice_loss_single(binary_seg_logits, binary_label)
            loss_fn = nn.CrossEntropyLoss()
            val_binary_loss = loss_fn(binary_seg_logits,binary_label)
            pix_embedding = net_output["instance_seg_logits"]
            val_instance_loss, _, _, _ = discriminative_loss(pix_embedding, instance_label, 3, 0.5, 1.5, 1.0, 1.0, 0.001)
            val_total_loss = val_binary_loss*1.0  + val_instance_loss * 0.1
            val_total_loss_list.append(val_total_loss.item())
            val_binary_loss_list.append(val_binary_loss.item())
            val_instance_loss_list.append(val_instance_loss.item())

            out = net_output["binary_seg_pred"]
            binary_label = binary_label.unsqueeze(1).type(torch.cuda.ByteTensor)
            pix_cls = out.masked_select(binary_label)
            val_acc.append(pix_cls.nonzero().size()[0]/pix_cls.size()[0])

        print("Epoch {} Validation Report".format(epoch+1),"total loss {:.5f}".format(np.mean(val_total_loss_list)),"binary loss {:.5f}".format(np.mean(val_binary_loss_list)),"instance loss {:.5f}".format(np.mean(val_instance_loss_list)),"valid accuracy {:.5f}".format(np.mean(val_acc)))
           # if ((epoch+1)%100 == 0):
        cv2.imwrite(os.path.join(im_path,str(epoch+1)+'binary_pred_label.png'), out[0].squeeze(0).cpu().numpy().transpose(0,1) * 255)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",help="train or test on lanenet model")
    parser.add_argument("--dataset",help="Directory of the AIC dataset")
    parser.add_argument("--save",help="Directory to save model checkpoint")
    parser.add_argument("--epochs",type=int,help="Training epochs")
    parser.add_argument("--pretrained",required = False,default=None,help="pretrained model path")
    parser.add_argument("--net",help = "backbone net")
    parser.add_argument("--image",help = "output val image")

#     parser.add_argument("--batch_size",help="Training or testing batch size")
#     parser.add_argument("--gpu",help="use gpu or not")
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    command = args.command
    save_path = args.save
    if command == 'train':
        train_net(args.dataset,save_path,args.image,args.epochs,args.pretrained,net_flag = args.net)

