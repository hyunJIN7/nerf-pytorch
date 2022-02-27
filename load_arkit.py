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
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def load_camera_pose(cam_pose_dir): # SyncedPose.txt
    if cam_pose_dir is not None and os.path.isfile(cam_pose_dir):
        pass
    else:
        raise FileNotFoundError("Given camera pose dir:{} not found"
                                .format(cam_pose_dir))

    pose = []
    def process(line_data_list):  #syncedpose.txt
        # imageNum(string) tx ty tz(m) qx qy qz qw
        line_data = np.array(line_data_list, dtype=float)
        # fid = line_data_list[0] #0부터
        trans = line_data[1:4]
        quat = line_data[4:]  #x,y,z,w 순?
        rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
                            # 여기선 (w,x,y,z) 순 인듯
        #TODO:check
        # rot_mat = rot_mat.dot(np.array([  #axis flip..?
        #     [1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, -1]
        # ]))
        # rot_mat = rotx(np.pi / 2) @ rot_mat #3D Rotation about the x-axis.
        # trans = rotx(np.pi / 2) @ trans
        trans_mat = np.zeros([3, 4])
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans
        trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
        pose.append(trans_mat)

    with open(cam_pose_dir, "r") as f:
        cam_pose_lines = f.readlines()
    for cam_line in cam_pose_lines:
        line_data_list = cam_line.split(" ")
        if len(line_data_list) == 0:
            continue
        process(line_data_list)

    return pose


def load_tum_data(basedir, min_angle=20,min_distance=0.1,ori_size=(1920, 1440), size=(640, 480)):

    # save image
    print('Extract images from video...')
    video_path = os.path.join(basedir, 'Frames.m4v')
    image_path = os.path.join(basedir, 'rgb')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        extract_frames(video_path, out_folder=image_path, size=size) #조건문 안으로 넣음


    # load intrin and extrin
    print('Load intrinsics and extrinsics')
    K = sync_intrinsics_and_poses(os.path.join(basedir, 'Frames.txt'), os.path.join(basedir, 'ARposes.txt'),
                            os.path.join(basedir, 'SyncedPoses.txt'))
    K[0,:] /= (ori_size[0] / size[0])             #image num(string) time(s) tx ty tz(m) qx qy qz qw
    K[1, :] /= (ori_size[1] / size[1])

    #quat -> rot
    all_cam_pose = load_camera_pose(os.path.join(basedir, 'SyncedPoses.txt'))


    """Keyframes selection"""
    all_ids = [0]
    last_pose = all_cam_pose[0]
    for i in range(len(all_cam_pose)):
        cam_intrinsic = K
        cam_pose = all_cam_pose[i]

        # translation->0.1m,rotation->15도 max 값 기준 넘는 것만 select
        angle = np.arccos(  # cos역함수  TODO: 여기 계산 과정 확인
            ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                [0, 0, 1])).sum())
        # extrinsice rotation 뽑아 inverse @  그 전 pose rotation @
        # rotation 사이 연산 후 accose 으로 각 알아내는
        dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
        # 기준값
        if angle > (min_angle / 180) * np.pi or dis > min_distance:
            all_ids.append(i)
            last_pose = cam_pose


    """final select image,poses"""
    image_dir = os.path.join(basedir, 'rgb')
    imgs = []
    poses = []
    for i in all_ids:
        image_file_name = os.path.join(image_dir, str(i).zfill(5) + '.jpg')
        imgs.append(imageio.imread(image_file_name))
        poses.append(all_cam_pose[i])
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    H, W = imgs[0].shape[:2]
    #TODO : .......how to find focal ....
    focal = K[0][0]

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
    n = poses.shape[0]  # count of image
    train_indexs = np.linspace(0, n, (int)(n * 0.8), endpoint=False, dtype=int)
    i_split.append(train_indexs)
    val_indexs = np.linspace(0, n, (int)(n * 0.2), endpoint=False, dtype=int)
    i_split.append(val_indexs)
    test_indexs = np.random.choice(n, (int)(n * 0.2))
    i_split.append(test_indexs)
    return imgs, poses, render_poses, [H, W, focal], K, i_split