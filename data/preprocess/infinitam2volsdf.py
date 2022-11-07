import collections
import argparse
import os
from xmlrpc.client import boolean

import numpy as np
import cv2
import matplotlib.pyplot as plt

from normalize_cameras import get_center_point, normalize_cameras, normalize_cameras_fake


def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing cameras')
    parser.add_argument('--poses', type=str)
    parser.add_argument('--infinitam_parameters', type=str)
    parser.add_argument('--no_normalization', action="store_true")
    parser.add_argument('--output_cameras_file', type=str, default="cameras.npz",
                        help='the output cameras file')
    parser.add_argument('--number_of_cams',type=int, default=-1,
                        help='Number of cameras, if -1 use all')

    args = parser.parse_args()

    image_poses = open(args.poses, "r").readlines()
    infinitam_parameters = open(args.infinitam_parameters, "r").readlines()

    cameras_npz_format = {}
    image_width = int(infinitam_parameters[0].split(" ")[0])
    image_height = int(infinitam_parameters[0].split(" ")[1])

    positions = []
    rotations = []

    ax = plt.figure().add_subplot(projection='3d')
    #plt.axis([-50, 0, -50, 0])

    for idx, pose in enumerate(image_poses):
        K = np.eye(3)

        K[0, 0] = float(infinitam_parameters[1].split(" ")[0])
        K[1, 1] = float(infinitam_parameters[1].split(" ")[1])
        K[0, 2] = float(infinitam_parameters[2].split(" ")[0])
        K[1, 2] = float(infinitam_parameters[2].split(" ")[1])

        # M = np.zeros((4,4))
        M = np.zeros((3,4))
        pose_data = [float(x) for x in pose.split()]
        M[0] = pose_data[:4]
        M[1] = pose_data[4:8]
        M[2] = pose_data[8:12]
        # M[3,3] = 1.0

        # M = np.linalg.inv(M)[:3, :]

        positions.append(M[:3, 3])
        rotations.append(rotation_angles(M[:3, :3], "xyz"))

        P=np.eye(4)
        P[:3,:] = M #K@M

        cameras_npz_format['world_mat_%d' % idx] = P
        
    x, y, z = np.array(positions)[:, 0], np.array(positions)[:, 1], np.array(positions)[:, 2]
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z))) 
    r0, r1, r2 = np.array(rotations)[:, 0] * np.pi / 180, np.array(rotations)[:, 1] * np.pi / 180, np.array(rotations)[:, 2] * np.pi / 180
    u = np.sin(r0 * x) * np.cos(r1 * y) * np.cos(r2 * z)
    v = -np.cos(r0* x) * np.sin(r1 * y) * np.cos(r2 * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(r0 * x) * np.cos(r1 * y) * np.sin(r2* z))
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=False)
    plt.show()

    np.savez(
            "cameras_before_normalization.npz",
            **cameras_npz_format)
    
    if args.no_normalization:
        normalize_cameras_fake("cameras_before_normalization.npz", args.output_cameras_file, args.number_of_cams, (image_width, image_height))
    else:
        normalize_cameras("cameras_before_normalization.npz", args.output_cameras_file, args.number_of_cams, (image_width, image_height))