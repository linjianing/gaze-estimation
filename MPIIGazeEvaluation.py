#!/usr/bin/python
#coding=utf-8

import math, shutil, os, time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='selfNetArc')
parser.add_argument('--out_person_num', type=int, required=False, default=0)
args=parser.parse_args()

from MPIIGazeData import MPIIGazeDataset
from modifiedITrackerModel import modifiedITrackerModel


from visdom import Visdom
vis = Visdom()

trainline = vis.line(Y=np.array([0]))
testline = vis.line(Y=np.array([0]))

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


# Change there flags to control what happens.
doLoad = False # Load checkpoint at the beginning
doTest = False # Only run test, no training

workers = 6
epochs = 20
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0
test_loss_output_dir = 'loss_data/MPIIGaze'

model_name = 'modifed_itracker_model'

def main():
    global args, best_prec1, weight_decay, momentum

    model = modifiedITrackerModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')


    root_dir = '/home/lin/Project/data_set/generateMPIIFaceEyeDataset/MPIIFaceGaze/'
    person_num = args.out_person_num
    dataTrain = MPIIGazeDataset(root_dir, person_num, split='train', imSize = imSize)
    dataVal = MPIIGazeDataset(root_dir, person_num, split='test', imSize = imSize)
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    criterion = nn.MSELoss().cuda()
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
	'''
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return
	'''
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    ''' 
    #used for visualize the training procedure
    time_p, train_losses, val_losses = [], [], []
    start_time = time.time()
    for epoch in range(epoch, epochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        
        # for visualize
        time_p.append(time.time() - start_time)
        train_losses.append(train_loss)
        val_losses.append(prec1)
        
        vis.line(X=np.column_stack((np.array([epoch]))),Y=np.column_stack((np.array([train_loss])))\
        , win=trainline, update='append', opts=dict(legend=['train_loss']))
        vis.line(X=np.column_stack((np.array([epoch]))),Y=np.column_stack((np.array([prec1])))\
        , win=testline, update='append', opts=dict(legend=['test_loss']))
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    
    loss_save_dir = os.path.join(test_loss_output_dir, '{:0>2}th_loss.txt'.format(person_num))
    if os.path.exists(test_loss_output_dir) is False:
        os.mkdir(test_loss_output_dir)
    with open(loss_save_dir,'w') as f:
        for value in val_losses:
            f.write(str(value) + '\n')


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (imFace, image_eye, headpose, gaze) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda(async=True)
        gaze = gaze.cuda(async=True)
        
        imFace = torch.autograd.Variable(imFace)
        gaze = torch.autograd.Variable(gaze)

        # compute output
        output = model(imFace, image_eye)

        loss = criterion(output, gaze)
        
        #losses.update(loss.data[0], imFace.size(0))

        euler_loss = computeEulerLoss(output.data, gaze.data)
        losses.update(euler_loss, 1) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        
    return losses.avg

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (imFace, image_eye, headpose, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda(async=True)
        gaze = gaze.cuda(async=True)
        
        imFace = torch.autograd.Variable(imFace, volatile = True)
        gaze = torch.autograd.Variable(gaze, volatile = True)

        # compute output
        output = model(imFace, image_eye)
        
        loss = criterion(output, gaze)
        
        euler_loss = computeEulerLoss(output.data, gaze.data)
        
        '''
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))
        '''
        losses.update(euler_loss, 1)
        #lossesLin.update(lossLin.data[0], imFace.size(0))

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {eulerLoss.val:.4f} ({eulerLoss.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   eulerLoss=losses))

    return losses.avg

CHECKPOINTS_PATH = './checkpoint/MPIIGaze'
if os.path.exists(CHECKPOINTS_PATH) is False:
    os.mkdir(CHECKPOINTS_PATH)

def load_checkpoint(filename='{}_{:0>2}thcheckpoint.pth.tar'.format(model_name,args.out_person_num)):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename='{}_{:0>2}thcheckpoint.pth.tar'.format(model_name,args.out_person_num)):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 7))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def computeEulerLoss(predict, target):
    data_x = (-1)*torch.cos(predict[:,0])*torch.sin(predict[:,1])
    data_y = (-1)*torch.sin(predict[:,0])
    data_z = (-1)*torch.cos(predict[:,0])*torch.cos(predict[:,1])
    norm_data = torch.sqrt(data_x*data_x + data_y*data_y + data_z*data_z)

    label_x = (-1)*torch.cos(target[:,0])*torch.sin(target[:,1])
    label_y = (-1)*torch.sin(target[:,0])
    label_z = (-1)*torch.cos(target[:,0])*torch.cos(target[:,1])
    norm_label = torch.sqrt(label_x*label_x + label_y*label_y + label_z*label_z)
    
    angle_value = (data_x*label_x+data_y*label_y+data_z*label_z) / (norm_data*norm_label)
    angle_error = (torch.acos(angle_value)*180)/np.pi
    
    #to delete the nan value
    angle_error = angle_error[1-torch.isnan(angle_error)]    
    angle_error = torch.mean(angle_error)
    return angle_error


if __name__ == "__main__":
    main()
    print('DONE')
