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


def extract_frames(video_path, out_folder, size):
    origin_size=[]
    """mp4 to image frame"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.resize(frame, size) #이미지 사이즈 변경에 따른 instrinsic 변화는 아래에 있음
        cv2.imwrite(os.path.join(out_folder, str(i).zfill(5) + '.jpg'), frame)
    return origin_size

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
def sync_time_image_pose(image_list,gt_list):#for tum
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

"""for arkit"""
def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    """Load camera intrinsics"""  # frane.txt -> camera intrinsics
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f:  # frame.txt 읽어서
        cam_intrinsic_lines = f.readlines()

    cam_intrinsics = []
    for line in cam_intrinsic_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_intrinsics.append([float(i) for i in line_data_list])
        # frame.txt -> cam_instrinsic
    K = np.array([
            [cam_intrinsics[0][2], 0, cam_intrinsics[0][4]],
            [0, cam_intrinsics[0][3], cam_intrinsics[0][5]],
            [0, 0, 1]
        ])

    """load camera poses"""  # ARPose.txt -> camera pose  gt
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()

    cam_poses = []
    for line in cam_pose_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])

    """ outputfile로 syncpose 맞춰서 내보냄  """
    lines = []
    ip = 0
    length = len(cam_poses)

    for i in range(len(cam_intrinsics)):
        while ip + 1 < length and abs(cam_poses[ip + 1][0] - cam_intrinsics[i][0]) < abs(
                cam_poses[ip][0] - cam_intrinsics[i][0]):
            ip += 1
        cam_pose = cam_poses[ip] #cam_poses[ip][:4] + cam_poses[ip][5:] + [cam_poses[ip][4]]
        line = [str(a) for a in cam_pose] #time,tx,ty,tz,qw,qx,qy,qz
        line[0] = str(i).zfill(5)  # name,tx,ty,tz,qw,qx,qy,qz
        lines.append(' '.join(line) + '\n')

    dirname = os.path.dirname(out_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_file, 'w') as f:
        f.writelines(lines)
    return K
