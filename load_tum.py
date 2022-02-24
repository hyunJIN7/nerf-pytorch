import pickle
from tqdm import tqdm

import os, imageio
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from transforms3d.quaternions import quat2mat
from load_data_helpers import *

np.random.seed(0)

def select_keyframe(img_list,pose_list, min_angle=15, min_distance=0.1):
    line_data = np.array(pose_list, dtype=float)# timestamp tx ty tz qx qy qz qw
    fid = line_data[0]  # 0부터 그냥 순번인듯
    trans = line_data[1:4]
    quat = line_data[4:]  # x,y,z,w
    rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist()) #(w,x,y,z)->rot mat

    return trans,rot_mat

def load_tum_data(basedir,tum_num_keyframe=120):
    #load -> sync -> selection (quater -> rotation)

    #instrin
    # K = [[535.4,0,320.1],
    #      [0,539.2,247,6],
    #      [0,0,1]]
    K = [[ 517.3,0,318.6],
         [0,516.5,255.3],
         [0,0,1]]


    #load image list(rgb.txt)
    image_text_file = os.path.join(basedir,'rgb.txt')
    """load camera ground trutth"""
    gt_file = os.path.join(basedir, 'groundtruth.txt')
    #TODO: change variable name
    image_list=dir2list(image_text_file) #timestamp filename
    gt_list=dir2list(gt_file) # timestamp tx ty tz qx qy qz qw

    """sync time between image and pose"""
    sync_gt_list = sync_time_image_pose(image_list,gt_list) #TODO: var name


    """quat -> rot"""
    line_data = np.array(sync_gt_list, dtype=float)  # timestamp tx ty tz qx qy qz qw
    trans = line_data[:,1:4] # tx ty tz
    quat = line_data[:,4:]  # qx qy qz qw

    """select key frame"""
    #indexs = select_keyframe(rot_mats)
    indexs = np.linspace(0, len(image_list), min(len(image_list),tum_num_keyframe),endpoint=False, dtype=int) #.astype(int)

    """final select image,poses"""
    imgs=[]
    poses=[]
    trans_mat = np.zeros([4,4])
    trans_mat[3,3] = 1
    for i in indexs:
        fname= os.path.join(basedir,'rgb',image_list[i][0]+'.png') #[1] 이 rgb/ .png 파일 명이지만 맨 마지막에 \n 붙어서 그냥 파임스템프로 함
        imgs.append(imageio.imread(fname))

        #quat(w,x,y,z) to rotation matrix
        rot_mat = quat2mat(np.append(quat[i][-1], quat[i][:3]))
        trans_mat[:3,3] = trans[i]
        trans_mat[:3,:3] = rot_mat
        poses.append(trans_mat)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)


    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    H, W = imgs[0].shape[:2]
    #focal = 535.4  #TODO : .......how to find focal ....
    focal =  517

    # TODO: change, Train, Val, Test ratio
    """
        여기 전체를 트레인 데이터 셋으로 하고 전체 개수에서 일정 개수만 랜덤으로 번호 골라서 val,test로 하게 코드 바꾸기 
    """
    # counts = [0]
    # n = poses.shape[0]
    # counts.append((int)(n*0.8))
    # counts.append(counts[-1] + (int)(n*0.15))
    # counts.append(n)
    # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    i_split = []
    n = poses.shape[0] # count of image
    train_indexs = np.linspace(0, n, (int)(n*0.8), endpoint=False, dtype=int)
    i_split.append(train_indexs)
    val_indexs = np.linspace(0, n, (int)(n*0.2), endpoint=False, dtype=int)
    i_split.append(val_indexs)
    test_indexs = np.random.choice(n, (int)(n*0.2))
    i_split.append(test_indexs)
    return imgs, poses, render_poses, [H, W, focal], K , i_split