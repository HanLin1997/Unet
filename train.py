import os, argparse, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import random_split, DataLoader

from collections import OrderedDict
from tqdm import tqdm

from model import NestedUNet
from dataset import MyDataset

random.seed(137)
np.random.seed(137)
torch.manual_seed(137)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(epoch, model, loss, accu):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'loss': loss,
        'accu': accu
    }
    torch.save(state, f'./save/{epoch}_{loss:.4f}_{accu:.4f}')

def IoU_score(predict, target, threshold=0.85):
    pred_binary = (predict > threshold).float()
    target_binary = (target > threshold).float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    iou = intersection / union if union > 0.01 else 0.0

    return iou.item()

def load_dataset(batchsize, num_workers=0):
    train_dataset = MyDataset()
    val_dataset = MyDataset()
    #tdataset, vdataset = random_split(dataset, [len(dataset)-80,80])
    #tdataset, vdataset = random_split(dataset, [16,16])
                                                
    train_loader = DataLoader(
        train_dataset,#Subset(tdataset, list(range(16))),
        batch_size=batchsize, 
        shuffle=True, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=num_workers,
        #collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batchsize, 
        shuffle=True, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=0,
        #collate_fn=collate_fn
    )
    return train_loader, val_loader

def load_network(class_num, lr, device, checkpoint_path=None):
    net = NestedUNet(class_num).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    iepoch = 0
    if not checkpoint_path is None:
        if os.path.exists(checkpoint_path):
            cp = torch.load(checkpoint_path)
            iepoch = int(cp['epoch'])+1

            #下面这一行是为了消除并行训练时，保存的网络参数名字会多出来一个 "xxx.module.xxxx"
            state_dict = OrderedDict( [(k.replace('module.',''),v) for k,v in cp['model'].items()] )
            net.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        net = torch.nn.DataParallel(net)

    return iepoch, net, optimizer

def train(epoch, net, loader, optimizer, criterion):
    losses = AverageMeter()
    accurs = AverageMeter()

    net.train()
    
    if isinstance(net, nn.DataParallel):
        device = next(net.module.parameters()).device
    else:
        device = next(net.parameters()).device
    
    total = len(loader)
    tmpl = '[{0}][{1}/{2}] LOSS:{loss.val:.4f}({loss.avg:.4f}) ACCURCY:{accur.val:.3f}({accur.avg:.3f})'
    msg = tmpl.format(epoch, 0, total, loss=losses, accur=accurs)
    with tqdm( total=total, desc=msg, ascii=True ) as bar:
        for i_batch, item in enumerate(loader):
            data, target = item

            input = data.to(device) / 255.0
            label = target.to(device)

            output = net(input.float())
            loss = criterion(output, label)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0, 2)
            optimizer.step()

            iou = 100*IoU_score(output, label)
            accurs.update(iou)

            msg = tmpl.format(epoch, i_batch, total, loss=losses, accur=accurs)
            bar.set_description(msg)
            bar.update()

    return losses.avg

def validate(net, loader, criterion):
    losses = AverageMeter()
    accurs = AverageMeter()

    net.eval()

    if isinstance(net,nn.DataParallel):
        device = next(net.module.parameters()).device
    else:
        device = next(net.parameters()).device

    total = len(loader)
    tmpl = '[Evaluate][{0}/{1}] LOSS:{loss.val:.4f}({loss.avg:.4f}) ACCURCY:{accur.val:.3f}({accur.avg:.3f})'
    msg = tmpl.format(0, total, loss=losses, accur=accurs)
    with tqdm( total=total,desc=msg, ascii=True ) as bar:
        for i_batch,item in enumerate(loader):
            data,target,target_lengths = item

            input = data.to(device) / 255.0
            label = target.to(device)
            
            with torch.no_grad():
                output = net(input.float())
                loss = criterion(output, label)
                losses.update(loss.item())
            
            accu = 100*IoU_score(output, label)
            accurs.update(accu)

            msg = tmpl.format(i_batch, total, loss=losses, accur=accurs)
            bar.set_description(msg)
            bar.update()
    return losses.avg


if __name__ == "__main__":
    epochs = 10
    batch_size = 16
    lr = 1e-4
    class_num = 1

    train_loader, valid_loader = load_dataset(batch_size)
    iepoch, net, optimizer = load_network(class_num, lr, device)#, checkpoint_path)

    criterion = nn.BCELoss()

    bestloss = np.inf

    for epoch in range(iepoch, epochs):
        vloss = train(epoch, net, train_loader, optimizer, criterion)
        vaccu = validate(net, valid_loader, criterion)
        if vloss < bestloss:
            save_model(epoch, net, vloss, vaccu)
            bestloss = vloss