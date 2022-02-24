import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from tqdm import tqdm

import os, imageio
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from transforms3d.quaternions import quat2mat
from bisect import bisect_left,bisect_right


trans_t = lambda t : torch.Tensor([ #z축 이동
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([  #rotation about x-axis
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([ #rotation about y-axis
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

                 #(angle, -30.0, 4.0)
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w  # @ : pytorch 에서 행렬곱
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def dir2list(dir):
    list = []
    assert os.path.isfile(dir), "rgb list info:{} not found".format(dir)
    with open(dir, "r") as f:
        list_lines = f.readlines()
    for line in list_lines[3:]:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        list.append([i for i in line_data_list])
    return list

"""sync time between image and pose"""
# def sync_time_image_pose_ver2(image_list,gt_list):
#     sync_gt_list =[]
#     for i,img_line in enumerate(image_list): # TODO : i = index 이렇게 더 효율있게 할 수 있 , i starts 0
#         timestamp = (float)(img_line[0])
#
#         index = 0
#         while index < len(gt_list)-2:
#             if abs(timestamp - (float)(gt_list[index][0]) ) >= abs(timestamp - (float)(gt_list[index+1][0])):
#                 index+=1;
#             else: break
#         sync_gt_list.append(gt_list[index])
#     return sync_gt_list
def sync_time_image_pose(image_list,gt_list):
    sync_gt_list =[]
    gt_list = np.array(gt_list).astype(np.float32)
    for i,img_line in enumerate(image_list): # TODO : i = index 이렇게 더 효율있게 할 수 있 , i starts 0
        timestamp = (float)(img_line[0])
        left_index = bisect_left(gt_list[:, 0], timestamp)
        if left_index >= gt_list.shape[0]-1 : left_index=gt_list.shape[0]-2
        if abs(timestamp - gt_list[left_index][0]) > abs(timestamp - gt_list[left_index + 1][0]):
            sync_gt_list.append(gt_list[left_index+1])
        else: sync_gt_list.append(gt_list[left_index])
    return sync_gt_list


#Neuralrecon baseline
def select_keyframe(rot_mats):
    indexs = [] #key frame select index
    count = 0
    last_pose = None
    min_angle = 15
    min_distance = 0.1
    for i,rot in tqdm(rot_mats, desc='Keyframes selection...'):
        if count == 0:
            indexs.append(id)
            last_pose = rot
            count += 1
        else:
            # translation->0.1m,rotation->15도 max 값 기준 넘는 것만 select
            angle = np.arccos(  # cos역함수  TODO: 여기 계산 과정 확인
                ((np.linalg.inv(rot) @ last_pose @ np.array([0, 0, 1]).T) * np.array([0, 0, 1])).sum())
            # extrinsice rotation 뽑아 inverse @  그 전 pose rotation @
            # rotation 사이 연산 후 accose 으로 각 알아내는
            dis = np.linalg.norm(rot - last_pose)
            # 기준값
            if angle > (min_angle / 180) * np.pi or dis > min_distance:
                indexs.append(i)
                last_pose = rot
    return indexs