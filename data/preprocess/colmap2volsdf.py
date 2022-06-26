import collections
import argparse
import os

import numpy as np
import cv2


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


def get_center_point(num_cams,cameras, img_shape):
    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, 1))
    camera_centers=np.zeros((3,num_cams))
    for i in range(num_cams):
        P0 = cameras['world_mat_%d' % i][:3, :]

        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:,i]=c[:3].flatten()

        v = np.linalg.inv(K) @ np.array([int(img_shape[0] / 2), int(img_shape[1] / 2), 1]) #(width/2, height/2, 1)
        v = v / np.linalg.norm(v)

        v=R[2,:]
        A[3 * i:(3 * i + 3), :3] = np.eye(3)
        A[3 * i:(3 * i + 3), 3 + i] = -v
        b[3 * i:(3 * i + 3)] = c[:3]

    soll= np.linalg.pinv(A) @ b

    return soll,camera_centers


def normalize_cameras(original_cameras_filename,output_cameras_filename,num_of_cameras, image_shape):
    cameras = np.load(original_cameras_filename)
    if num_of_cameras==-1:
        all_files=cameras.files
        maximal_ind=0
        for field in all_files:
            maximal_ind=np.maximum(maximal_ind,int(field.split('_')[-1]))
        num_of_cameras=maximal_ind+1
    soll, camera_centers = get_center_point(num_of_cameras, cameras, image_shape)

    center = soll[:3].flatten()

    max_radius = np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = center[0]
    normalization[1, 3] = center[1]
    normalization[2, 3] = center[2]

    normalization[0, 0] = max_radius / 3.0
    normalization[1, 1] = max_radius / 3.0
    normalization[2, 2] = max_radius / 3.0

    cameras_new = {}
    for i in range(num_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = cameras['world_mat_%d' % i].copy()
    np.savez(output_cameras_filename, **cameras_new)


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
 