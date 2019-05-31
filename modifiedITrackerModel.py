import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

'''
Pytorch model for the iTracker.

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


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class modifiedItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(modifiedItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # add the special weight layer
        self.special_weight = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x_sep = self.special_weight(x)
        x = x*x_sep
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class modifiedFaceImageModel(nn.Module):
    
    def __init__(self):
        super(modifiedFaceImageModel, self).__init__()
        self.conv = modifiedItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class modifiedITrackerModel(nn.Module):


    def __init__(self):
        super(modifiedITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = modifiedFaceImageModel()
        #self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(12*12*64, 64),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )
        #self.theta = nn.Linear(64,1)
        #self.phi = nn.Linear(64,1)

    def forward(self, faces, eyes):
        # Eye nets 
        xEyes = self.eyeModel(eyes)
        xEyes = self.eyesFC(xEyes)
        # Cat and FC

        # Face net
        xFace = self.faceModel(faces)

        #xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace), 1)
        x = self.fc(x)
        #theta = self.theta(x)
        #phi = self.phi(x)

        return x

class grayModifiedITrackerModel(nn.Module):


    def __init__(self):
        super(grayModifiedITrackerModel, self).__init__()
        #self.eyeModel = ItrackerImageModel()
        self.faceModel = modifiedFaceImageModel()

        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )
        #self.theta = nn.Linear(64,1)
        #self.phi = nn.Linear(64,1)

    def forward(self, faces):
        # Eye nets 
        #xEyeL = self.eyeModel(eyesLeft)
        #xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        #xEyes = torch.cat((xEyeL, xEyeR), 1)
        #xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)

        #xGrid = self.gridModel(faceGrids)

        # Cat all
        #x = torch.cat((xEyes, xFace), 1)
        x = self.fc(xFace)
        #theta = self.theta(x)
        #phi = self.phi(x)

        return x


class binnedModel(nn.Module):


    def __init__(self):
        super(binnedModel, self).__init__()
        #self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        #self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*12*12*64, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            #nn.Linear(64, 2),
            )
        self.theta = nn.Linear(64,67)
        self.phi = nn.Linear(64,67)

    def forward(self, faces, eyesLeft, eyesRight):

        # Face net
        xFace = self.faceModel(faces)

        theta_part = self.fc(xFace)
        phi_part = self.fc(xFace)
        theta = self.theta(theta_part)
        phi = self.phi(phi_part)

        return theta, phi


class gazelayer(nn.Module):
    
    def __init__(self):
        super(gazelayer, self).__init__()
        self.premodel = modifiedITrackerModel()
        self.gz = nn.Sequential(
            nn.Linear(5,10),
            nn.ReLU(inplace=True),
            nn.Linear(10,2),
        )

    def forward(self, faces, eyesLeft, eyesRight, headposes):
        pre_gaze = self.premodel(faces, eyesLeft, eyesRight)
        cat = torch.cat((pre_gaze, headposes), 1)
        gaze = self.gz(cat)
        return gaze


class EyePatchModel(nn.Module):


    def __init__(self):
        super(EyePatchModel, self).__init__()
        self.features = ItrackerImageModel()

        self.eyePatchFc = nn.Sequential(
            nn.Linear(10*2*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64,2)
            )
        
        '''
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )
        '''
    def forward(self, faces, eyesLeft, eyesRight):
        xFace = self.features(faces)
        x = self.eyePatchFc(xFace)

        return x

class EyePatchWithHpModel(nn.Module):


    def __init__(self):
        super(EyePatchWithHpModel, self).__init__()
        self.features = ItrackerImageModel()

        self.eyePatchFc = nn.Sequential(
            nn.Linear(10*2*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )
        
        self.gaze = nn.Sequential(
            nn.Linear(67, 2)
        )
        '''
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )
        '''
    def forward(self, faces, eyesLeft, eyesRight, headposes):
        xFace = self.features(faces)
        x = self.eyePatchFc(xFace)
        x = torch.cat((x, headposes), 1)
        x = self.gaze(x)

        return x