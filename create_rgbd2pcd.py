import argparse
import sys
import os

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
from PIL import Image

focalLength = 938.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000

"""
https://ichi.pro/ko/python-eulo-3d-pointeu-keullaudeu-cheoli-al-abogi-106733593573607
"""

def generate_pointcloud(rgb_file,depth_file,ply_file):

    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file).convert('I')

    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            print(Z)
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append(np.vstack(X,Y,Z).T)

    # mean_Z = np.mean(points,axis=0)[2]
    # spatial_query = point_cloud[abs(point_cloud[:, 2] - mean_Z) < 1]
    # xyz = spatial_query[:, :3]
    # rgb = spatial_query[:, 3:]
    # ax = plt.axes(projection='3d')
    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255, s=0.01)
    # plt.show()


if __name__=='__main__':
    generate_pointcloud()
