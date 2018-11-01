import torch
import torch.nn as nn
import torch.nn.functional as F

def bootstrapped_cross_entropy_single(input, target, K, weight=None, size_average=True):
    n, c, h, w = input.size()
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    # print (input.size())
    # print (target.size())
    loss = F.cross_entropy(input, target, weight=weight, reduce=False, size_average=False, ignore_index=250)
    # print ('loss', loss.size())
    topk_loss, _ = loss.topk(K)
    reduced_topk_loss = topk_loss.sum() / K
    return reduced_topk_loss

def bootstrapped_cross_entropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    batch_size = n
    loss = 0.0

    # Bootstrap from each image not entire batch
    K = h * w // 2
    for i in range(1, batch_size):
        loss += bootstrapped_cross_entropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

def one_hot(x, num_class):
    n, h, w = x.size()
    x = x.view(n, 1, h, w)
    x_onehot = torch.FloatTensor(n, num_class, h, w).cuda()
    x_onehot.zero_()
    x_onehot.scatter_(1, x, 1)
    return x_onehot

def dice_loss_single(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    # mask = mask.contiguous().view(mask.size()[0], -1)
    # input = input * mask
    # target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 1e-3
    c = torch.sum(target * target, 1) + 1e-3
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def dice_loss(input, target):
    input = F.sigmoid(input)
    n, c, h, w = input.size()

    target = one_hot(target, c)

    loss = 0.0
    for i in range(1, c):
        loss += dice_loss_single(input[:, i, :, :], target[:, i, :, :])
    return loss
