#!/usr/bin/python
#coding=utf-8

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
    read the original image and landmarks and normaliza then return the pytorch version of 
    data input.
    need two parameters: 
    1. dataroot: recording the original data information (equal to annotationroot) 
    2. imageroot: recording the image data information

    data need to be processed:
    1. sessions correpond to specific person
    2. groups correspond to specific people
    3. train and test data correspond to specific groups

    train and test dict should contain the following information:
    1. groups: a dict contains the people in corresponding group
    2. annotations contains the headpose 
'''

'''
    root_dir = '/media/lin/新加卷/ubuntu_file/eyediap/EYEDIAP/img_limit_gaze_angle'
'''

def read_calib_param(calibration_file):
    with open(calibration_file, 'r') as f:
    	lines = f.readlines()
    camera_param = {}
    intrinsic_para = lines[3:6]
    R_param = lines[7:10]
    T_param = lines[11:]
    intrinsics = np.zeros((3,3))
    R = np.zeros((3,3))
    T = np.zeros((3,1))
    for index in range(len(intrinsic_para)):
		intrinsics[index,:] = np.array([float(x) for x in intrinsic_para[index].rstrip().split(';')])
    for index in range(len(R_param)):
		R[index,:] = np.array([float(x) for x in R_param[index].rstrip().split(';')])
    for index in range(len(T_param)):
		T[index,:] = np.array([float(x) for x in T_param[index].rstrip().split(';')])
    camera_param['intrisics'] = intrinsics
    camera_param['R'] = R
    camera_param['T'] = T
    return camera_param

def read_face_shape_param(face_location_file):
    with open(face_location_file, 'r') as f:
        lines = f.readlines()
    face_shape = {}
    for index in range(len(lines)):
        cols = lines[index].rstrip().split(' ')
        img_name = cols[0]
        face_landmark = np.array([float(x) for x in cols[1:]])
        face_shape[img_name] =face_landmark
    return face_shape

def read_eyelocation_param_3D(eyelocation_file):
    with open(eyelocation_file, 'r') as f:
        lines = f.readlines()[1:]
    eye_positions = np.zeros((len(lines), 6))
    for index in range(len(lines)):
        eye_positions[index] = np.array([float(x) for x in lines[index].rstrip().split(';')[-6:]])
    return eye_positions

def read_eyelocation_param_2D(eyelocation_file):
    with open(eyelocation_file, 'r') as f:
        lines = f.readlines()[1:]
    eye_positions = np.zeros((len(lines), 4))
    for index in range(len(lines)):
        eye_positions[index] = np.array([float(x) for x in lines[index].rstrip().split(';')[1:5]])
    return eye_positions


def normalizeImg(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    '''
        in order to overcome the difference of the camera internal parameters 
    and the the distance from face center to the camera optical
    '''
    #new virtual camera
    focal_new = 960.0
#    distance_new=300.0 for 448*448
    distance_new = 0.6

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
    #htnew = np.dot(cnvMat, target_3D)

    #gaze vector normalization
    #gcnew = np.dot(cnvMat, gc)
    gv = gc - target_3D
    gvnew = np.dot(rotMat, gv)
    gvnew = gvnew / np.linalg.norm(gvnew)

    return img_warped, hrnew, gvnew


def getDataAndParameters(session_list, img_root, data_root):
    '''
        img_root = '/media/lin/新加卷/ubuntu_file/eyediap/EYEDIAP/img_limit_gaze_angle'
        data_root = '/media/lin/新加卷/ubuntu_file/eyediap/EYEDIAP/Data'   used for annotations
    
    '''

    img_file_list = []
    annotations = {}

    ##read the correspond annotation and the get the relationship of person and session
    ##useful annotation list: 1. head pose; 2. gaze; 3.eye location in img
    for session in session_list:

    #read annotations
        session = session.rstrip()
        annotation_root = os.path.join(data_root, session)

        head_pose_file = os.path.join(annotation_root, 'head_pose.txt')
        eye_location_file = os.path.join(annotation_root, 'eye_tracking.txt')
        screen_target_file = os.path.join(annotation_root, 'screen_coordinates.txt')
        camera_calib_file = os.path.join(annotation_root, 'rgb_vga_calibration.txt')

        #headpose.shape [frame_num, 0-8 rotation 9-11 translation]
        head_pose_vals = np.loadtxt(head_pose_file, skiprows=1, delimiter=';')[:, 1:]
        head_pose_append = np.zeros((1,head_pose_vals.shape[1]))
        head_pose_vals = np.row_stack((head_pose_vals,head_pose_append))
        eye_locations_3D = read_eyelocation_param_3D(eye_location_file)
        eye_locations_3D_append = np.zeros((1,eye_locations_3D.shape[1]))
        eye_locations_3D = np.row_stack((eye_locations_3D,eye_locations_3D_append))
        eye_locations_2D = read_eyelocation_param_2D(eye_location_file)
        eye_locations_2D_append = np.zeros((1,eye_locations_2D.shape[1]))
        eye_locations_2D = np.row_stack((eye_locations_2D,eye_locations_2D_append))
        screen_targets = np.loadtxt(screen_target_file, skiprows=1, delimiter=';')[:, -3:]
        screen_targets_append = np.zeros((1,screen_targets.shape[1]))
        screen_targets = np.row_stack((screen_targets,screen_targets_append))
        camera_calib = read_calib_param(camera_calib_file)

        #get the shape information of face
        face_shape_file = os.path.join(img_root, session, 'face_landmark.txt')
        face_shape = read_face_shape_param(face_shape_file)

        param_dict = {}
        param_dict['face_shape'] = face_shape
        param_dict['headpose'] = head_pose_vals
        param_dict['eyelocation3D'] = eye_locations_3D
        param_dict['eyelocation2D'] = eye_locations_2D
        param_dict['screentarget'] = screen_targets
        param_dict['cameracalib'] = camera_calib
        annotations[session] = param_dict

    #generate image list
        img_list = face_shape.keys()
        img_file_list.extend(img_list)

    return img_file_list, annotations


def getTrainDict(groupout, groups, img_root, data_root):
    '''
        get the train dict
    '''
    session_list = os.listdir(img_root)
    train_people = []
    train_session = []
    for gn in groups.keys():
        if groupout != gn:
            train_people.extend(groups[gn])
    for session in session_list:
        if int(session.split('_')[0]) in train_people:
            train_session.append(session)

    img_file_list, annotations = getDataAndParameters(train_session, img_root, data_root)

    return img_file_list, annotations

def getTestDict(groupout, groups, img_root, data_root):
    '''
        get the test list
    '''
    session_list = os.listdir(img_root)
    test_people = []
    test_session = []
    for gn in groups.keys():
        if groupout == gn:
            test_people.extend(groups[gn])
    for session in session_list:
        if int(session.split('_')[0]) in test_people:
            test_session.append(session)

    img_file_list, annotations = getDataAndParameters(test_session, img_root, data_root)

    return img_file_list, annotations


class EyeDiapDataset(Dataset):
    def __init__(self, data_root, img_root, groupout, groups, split ='train', imSize=(224,224)):
        '''
            groups: a dict with key equals group id and values equal corresponding people numbers
            groupout: a int indicate the number of group as a test set
        '''
        
        self.imSize = imSize
        self.data_root = data_root
        self.img_root = img_root
        self.groups = groups
        self.groupout = groupout
        

        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.5),
            #transforms.RandomAffine(degrees=0,translate=(0.05,0.05)),
            #transforms.RandomCrop((200,200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            #transforms.ColorJitter(brightness=0.5),
            #transforms.RandomAffine(degrees=0,translate=(0.05,0.05)),
            #transforms.RandomCrop((200,200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            #transforms.ColorJitter(brightness=0.5),
            #transforms.RandomAffine(degrees=0,translate=(0.05,0.05)),
            #transforms.RandomCrop((200,200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        
        if split =='train':
            self.img_file_list, self.annotations = getTrainDict(groupout, groups, img_root, data_root)
        else:
            self.img_file_list, self.annotations = getTestDict(groupout, groups, img_root, data_root)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    
    def __getitem__(self, idx):
        img_file = self.img_file_list[idx]
        img_ori = cv2.imread(img_file)
    
        #hist equalization 
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        img_equ = self.clahe(img_ori)

        img_index = int(img_file.split('/')[-1].split('.')[0])
        session = img_file.split('/')[-2]
        hR = self.annotations[session]['headpose'][img_index][:9].reshape(3,3)

        eye_center_left_2D = self.annotations[session]['eyelocation2D'][img_index][:2]
        eye_center_right_2D = self.annotations[session]['eyelocation2D'][img_index][2:]

        eye_right_width = 0.6*np.linalg.norm(eye_center_right_2D - eye_center_left_2D)
        eye_left_width = eye_right_width

        ##get the correponding eye area from the image
        eye_left_start_x =int(eye_center_left_2D[0]-eye_left_width/2)
        eye_left_start_x = eye_left_start_x if eye_left_start_x>0 else 0 
        eye_left_start_y = int(eye_center_left_2D[1]-eye_left_width/2)
        eye_left_start_y = eye_left_start_y if eye_left_start_y>0 else 0
        eye_left = img_equ[eye_left_start_y:int(eye_left_start_y+eye_left_width)\
        ,eye_left_start_x:int(eye_left_start_x+eye_left_width)].copy()

        eye_right_start_x =int(eye_center_right_2D[0]-eye_right_width/2)
        eye_right_start_x = eye_right_start_x if eye_right_start_x>0 else 0 
        eye_right_start_y = int(eye_center_right_2D[1]-eye_right_width/2)
        eye_right_start_y = eye_right_start_y if eye_right_start_y>0 else 0
        eye_right = img_equ[eye_right_start_y:int(eye_right_start_y+eye_right_width)\
        ,eye_right_start_x:int(eye_right_start_x+eye_right_width)].copy()

        '''
        left_b, left_g, left_r = cv2.split(eye_left)
        left_b_hist = cv2.equalizeHist(left_b)
        left_g_hist = cv2.equalizeHist(left_g)
        left_r_hist = cv2.equalizeHist(left_r)
        eye_left = cv2.merge((left_b_hist, left_g_hist, left_r_hist))
        
        right_b, right_g, right_r = cv2.split(eye_right)
        right_b_hist = cv2.equalizeHist(right_b)
        right_g_hist = cv2.equalizeHist(right_g)
        right_r_hist = cv2.equalizeHist(right_r)
        eye_right = cv2.merge((right_b_hist, right_g_hist, right_r_hist))
        '''

        eye_left_resized = cv2.resize(eye_left, self.imSize, interpolation=cv2.INTER_CUBIC)
        eye_right_resized = cv2.resize(eye_right, self.imSize, interpolation=cv2.INTER_CUBIC)
        
        #cv2.imshow('eye_left_resized', eye_left_resized)
        #cv2.imshow('eye_right_resized', eye_right_resized)

        image_left = Image.fromarray(cv2.cvtColor(eye_left_resized, cv2.COLOR_BGR2RGB))
        image_right = Image.fromarray(cv2.cvtColor(eye_right_resized, cv2.COLOR_BGR2RGB))

        ##face part
        #get the face center in the original camera 
        #gaze_target = screen_target 
        #gaze direction = gaze_traget - gaze_origin(np.dot(hR, eyeballcenter)+ht)
        eye_center_left_3D = self.annotations[session]['eyelocation3D'][img_index][:3]
        eye_center_right_3D = self.annotations[session]['eyelocation3D'][img_index][3:]
        face_center = 0.5*(eye_center_left_3D+eye_center_right_3D)
        gaze_target = self.annotations[session]['screentarget'][img_index]
        external_R = self.annotations[session]['cameracalib']['R']
        external_T = self.annotations[session]['cameracalib']['T']
        face_center_wc = (np.dot(np.linalg.inv(external_R), face_center.reshape(3,1)-external_T)).flatten()

        #use the shape clue to change the face image
        face_shape = self.annotations[session]['face_shape'][img_file]
        x = float(face_shape[4] - face_shape[2])
        y = float(face_shape[5] - face_shape[3])
        angle = math.atan(y/x)
        eye_center = (eye_center_left_2D+eye_center_right_2D)/2
        rotation = cv2.getRotationMatrix2D((eye_center[0],eye_center[1]), angle/np.pi*180, 1.0)
        img_equ = cv2.warpAffine(img_equ, rotation, img_equ.shape[:2])
        hR = cv2.Rodrigues(hR)[0][:,0]
        #set the roll angle to 0
        hR[2] = 0
        hR = cv2.Rodrigues(hR)[0]

        #normalization 
        img, headpose, norm_gaze = normalizeImg(img_equ,face_center_wc, hR, gaze_target, \
        self.imSize, self.annotations[session]['cameracalib']['intrisics'])
        
        #put the image into the data set
        eye_left_resized = Image.fromarray(eye_left_resized.astype('uint8'))
        eye_right_resized = Image.fromarray(eye_right_resized.astype('uint8'))
        img = Image.fromarray(img.astype('uint8'))
        eyeL = self.transformEyeL(eye_left_resized)
        eyeR = self.transformEyeR(eye_right_resized)
        face = self.transformFace(img)

        image_fee = torch.cat((eyeL,eyeR,face),0)

        headposes = np.array(headpose[:,0],dtype=np.float32)

        # rotate the gaze angle by the same way
        gaze_rot = cv2.Rodrigues(np.array([0,0,angle]))[0] 
        norm_gaze = np.dot(gaze_rot, norm_gaze)

        #cv2.imshow('image_warped', img)
        #print 'warped image.shape:{}'.format(img.shape)

        #convert the gaze direction in the camera coordinate system to the angle
        #in the polar coordinate system
        #make the norm_gaze by external_R * norm_gaze
        #norm_gaze = np.dot(external_R, norm_gaze)
        #make the norm_gaze by original gaze vector
        #norm_gaze = (gaze_target - face_center)/np.linalg.norm(gaze_target - face_center)
        gaze_theta = math.asin((-1)*norm_gaze[1]) #vertical gaze angle
        gaze_phi = math.atan2((-1)*norm_gaze[0], (-1)*norm_gaze[2]) #horizontal gaze angle
        gaze = np.array([gaze_theta, gaze_phi],np.float32) 
    
        return image_fee, headposes, gaze


    def __len__(self):
        return len(self.img_file_list)