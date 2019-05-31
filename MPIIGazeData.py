import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
import math

'''
    read the original image and landmarks and normaliza than return the pytorch version of 
    data input.
'''

'''
    root_dir = '/home/lin/Project/data_set/generateMPIIFaceEyeDataset/MPIIFaceGaze/'
'''

def normalizeImg(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    '''
        inorder to overcome the difference of the camera internal parameters 
    and the the distance from face center to the camera optical
    '''
    #new virtual camera
    focal_new = 960.0
#    distance_new=300.0 for 448*448
    distance_new = 600.0

    distance = np.linalg.norm(target_3D)
    z_scale = distance_new/distance
    cam_new = np.array([[focal_new, 0.0, roiSize[0]/2],\
    [0.0, focal_new, roiSize[1]/2], [0.0, 0.0, 1.0]], np.float32)
    scaleMat = np.array([[1.0, 0.0, 0.0],\
    [0.0, 1.0, 0.0],[0.0, 0.0, z_scale]], np.float32)
    hRx = hR[:, 0]
    forward = target_3D/distance
    down = np.cross(forward, hRx)
    down = down/np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right/np.linalg.norm(right)
    rotMat = np.c_[right, down, forward]
    rotMat = rotMat.transpose()
    warpMat = np.dot(np.dot(cam_new, scaleMat), \
    np.dot(rotMat, np.linalg.inv(cameraMatrix)))
    img_warped = cv2.warpPerspective(inputImg, warpMat, roiSize)
    #img_warped = np.transpose(img_warped, (1,0,2))

    #rotation normalization
    cnvMat = np.dot(scaleMat, rotMat)
    hRnew = np.dot(cnvMat, hR)
    hrnew = cv2.Rodrigues(hRnew)[0]
    htnew = np.dot(cnvMat, target_3D)

    #gaze vector normalization
    gcnew = np.dot(cnvMat, gc)
    gvnew = gcnew - htnew
    gvnew = gvnew / np.linalg.norm(gvnew)

    return img_warped, hrnew, gvnew

def readAnnotations(pn, root_dir):
    '''
        read annotations from files
    '''
    annotation_dict = {}
    annotation_path = os.path.join(root_dir, 'p{:0>2}'.format(pn), 'p{:0>2}.txt'.format(pn))
    with open(annotation_path, 'r') as f:
        annotation = f.readlines()
    for line in annotation:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        img_name = os.path.join(root_dir, 'p{:0>2}'.format(pn), img_name)
        annotation_dict[img_name] = line[1:]
    return annotation_dict

def getTrainDict(personout, root_dir):
    '''
        get the train list
        input(int): the person number correspond to test set
    '''
    trainDict = {}
    calibrations = {}
    for pn in range(15):
        if pn == personout:
            continue
        annotations = readAnnotations(pn, root_dir)
        trainDict.update(annotations)
        calibration_path = os.path.join(root_dir, 'p{:0>2}'.format(pn), 'Calibration/Camera.mat')
        cameraCalib = sio.loadmat(calibration_path)
        calibrations[pn] = cameraCalib
    return trainDict, calibrations

def getTestDict(personout, root_dir):
    '''
        get the test list
        input(int): the person number correspond to test set
    '''
    testDict = readAnnotations(personout, root_dir)
    calibrations = {}
    calibration_path = os.path.join(root_dir, 'p{:0>2}'.format(personout), 'Calibration/Camera.mat')
    cameraCalib = sio.loadmat(calibration_path)
    calibrations[personout] = cameraCalib
    return testDict, calibrations


class MPIIGazeDataset(Dataset):
    def __init__(self, root_dir, personout, split ='train', imSize=(224,224)):

        self.imSize = imSize
        self.data_root = root_dir
        self.personout = personout
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        
        if split =='train':
            self.annotations, self.calibrations = getTrainDict(personout, root_dir)
        else:
            self.annotations, self.calibrations = getTestDict(personout, root_dir)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    
    def __getitem__(self, idx):
        img_list = self.annotations.keys()
        image_name = img_list[idx]
        person = int(image_name.split('/')[-3][1:])
        annotation = self.annotations[image_name]

        img_ori = cv2.imread(image_name)

        #hist equalization 
        b,g,r = cv2.split(img_ori)
        b_eq_hist = self.clahe.apply(b)
        g_eq_hist = self.clahe.apply(g)
        r_eq_hist = self.clahe.apply(r)
        img = cv2.merge((b_eq_hist, g_eq_hist, r_eq_hist))
        #print 'original image.shape:{}'.format(img.shape)

        #print 'image read from img_path.type:{}'.format(type(img))
        cameraCalib = self.calibrations[person]

        #get head pose
        headpose_hr = np.array(annotation[14:17], np.float32)
        #print 'orgin head rotation:{}'.format(headpose_hr)
        headpose_ht = np.array(annotation[17:20], np.float32)
        headpose_ht =  headpose_ht.reshape(3,1)
        #print headpose_ht
        hR = cv2.Rodrigues(headpose_hr)[0]

        
        ##eye part
        #get the eye center in the original camera
        eye_center_right = 0.5*(np.array(annotation[8:10], np.float32)\
        +np.array(annotation[6:8], np.float32))
        eye_center_left = 0.5*(np.array(annotation[4:6], np.float32)\
        +np.array(annotation[2:4], np.float32))

        #cv2.circle(img, (eye_center_right[0], eye_center_right[1]), 5, (255,255,255), -1)
        #cv2.circle(img, (eye_center_left[0], eye_center_left[1]), 5, (255,255,255), -1)


        eye_right_width = 1.7*np.linalg.norm(np.array(annotation[8:10], np.float32)\
        -np.array(annotation[6:8], np.float32))
        eye_left_width = 1.7*np.linalg.norm(np.array(annotation[4:6], np.float32)\
        -np.array(annotation[2:4], np.float32))

        #get the correponding area from the image
        eye_left_start_x =int(eye_center_left[0]-eye_left_width/2)
        eye_left_start_x = eye_left_start_x if eye_left_start_x>0 else 0 
        eye_left_start_y = int(eye_center_left[1]-eye_left_width/2)
        eye_left_start_y = eye_left_start_y if eye_left_start_y>0 else 0
        eye_left = img[eye_left_start_y:int(eye_left_start_y+eye_left_width)\
        ,eye_left_start_x:int(eye_left_start_x+eye_left_width),:].copy()

        eye_right_start_x =int(eye_center_right[0]-eye_right_width/2)
        eye_right_start_x = eye_right_start_x if eye_right_start_x>0 else 0 
        eye_right_start_y = int(eye_center_right[1]-eye_right_width/2)
        eye_right_start_y = eye_right_start_y if eye_right_start_y>0 else 0
        eye_right = img[eye_right_start_y:int(eye_right_start_y+eye_right_width)\
        ,eye_right_start_x:int(eye_right_start_x+eye_right_width),:].copy()

        #cv2.imshow('eye left origin', eye_left)
        #cv2.imshow('eye right origin', eye_right)

        eye_left_resized = cv2.resize(eye_left, self.imSize ,interpolation=cv2.INTER_CUBIC)
        eye_right_resized = cv2.resize(eye_right, self.imSize, interpolation=cv2.INTER_CUBIC)

        #image_left = eye_left_resized.transpose(2,0,1) 
        #image_right = eye_right_resized.transpose(2,0,1) 
        
        image_left = Image.fromarray(cv2.cvtColor(eye_left_resized, cv2.COLOR_BGR2RGB))
        image_right = Image.fromarray(cv2.cvtColor(eye_right_resized, cv2.COLOR_BGR2RGB))

        image_left = self.transformEyeL(image_left)
        image_right = self.transformEyeR(image_right)

        image_eye = torch.cat((image_left, image_right), 0)		
        #cv2.imshow('left eye part', eye_left_resized)
        #cv2.imshow('right eye part', eye_right_resized)

        ##face part
        #get the face center in the original camera
        face_center = np.array(annotation[20:23], np.float32)
        gaze_target = np.array(annotation[23:26], np.float32)

        #normalization 
        img, headpose, norm_gaze = normalizeImg(img,face_center, hR, gaze_target, \
        self.imSize, cameraCalib['cameraMatrix'])

        #put the image into the data set
        #image_face = img.transpose(2,0,1)
        image_face = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_face = self.transformFace(image_face)
        headposes = headpose[:,0]

        #cv2.imshow('image_warped', img)
        #print 'warped image.shape:{}'.format(img.shape)

        #convert the gaze direction in the camera coordinate system to the angle
        #in the polar coordinate system
        gaze_theta = math.asin((-1)*norm_gaze[1]) #vertical gaze angle
        gaze_phi = math.atan2((-1)*norm_gaze[0], (-1)*norm_gaze[2]) #horizontal gaze angle
        gaze = np.array([gaze_theta, gaze_phi], dtype=np.float32) 

        return image_face, image_eye, headposes, gaze


    def __len__(self):
        return len(self.annotations.keys())
