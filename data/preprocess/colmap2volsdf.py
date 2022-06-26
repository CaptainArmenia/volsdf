import collections
import argparse
import os

import numpy as np
import cv2

from normalize_cameras import get_center_point, normalize_cameras


BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing cameras')
    parser.add_argument('--colmap_project_dir', type=str)
    parser.add_argument('--output_cameras_file', type=str, default="cameras.npz",
                        help='the output cameras file')
    parser.add_argument('--number_of_cams',type=int, default=-1,
                        help='Number of cameras, if -1 use all')

    args = parser.parse_args()

    cameras=read_cameras_text(os.path.join(args.colmap_project_dir, "cameras.txt"))
    images=read_images_text(os.path.join(args.colmap_project_dir, "images.txt"))

    cameras_npz_format = {}
    image_width = cameras[1].width
    image_height = cameras[1].height

    for ii in sorted(images.keys()):
        cur_image=images[ii]
        cur_camera_id = cur_image[3]
        K = np.eye(3)
        cur_camera = cameras[cur_camera_id]

        if not cur_camera.model in ["OPENCV", "PINHOLE"]:
            raise Exception("Invalid camera model. Select PINHOLE or OPENCV camera model at colmap for reconstruction.")

        K[0, 0] = cur_camera.params[0]
        K[1, 1] = cur_camera.params[1]
        K[0, 2] = cur_camera.params[2]
        K[1, 2] = cur_camera.params[3]

        M=np.zeros((3,4))
        M[:,3]=cur_image.tvec
        M[:3,:3]=qvec2rotmat(cur_image.qvec)

        P=np.eye(4)
        P[:3,:] = K@M
        cameras_npz_format['world_mat_%d' % int(cur_image.name.split(".")[0])] = P
        
    np.savez(
            "cameras_before_normalization.npz",
            **cameras_npz_format)
        
    normalize_cameras("cameras_before_normalization.npz", args.output_cameras_file, args.number_of_cams, (image_width, image_height))
 