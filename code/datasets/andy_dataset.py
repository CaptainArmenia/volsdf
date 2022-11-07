import os
import torch
import numpy as np
import pickle

import utils.general as utils
from utils import rend_util


class AndyDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_data_file = '{0}/cameras.pkl'.format(self.instance_dir)
        self.image_data = pickle.load(open(image_data_file, "rb"))
        self.n_images = len(self.image_data.keys())
        self.scale_mat = None

        for image_id in self.image_data.keys():
            world_mat = self.image_data[image_id]["world_mat"].astype(np.float32)
            scale_mat = self.image_data[image_id]["scale_mat"].astype(np.float32)

            if self.scale_mat is None:
                self.scale_mat = scale_mat

            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.image_data[image_id]["pose"] = torch.from_numpy(pose).float()
            self.image_data[image_id]["intrinsics"] = torch.from_numpy(intrinsics).float()
            rgb = rend_util.load_rgb(os.path.join(image_dir, self.image_data[image_id]["image_file"]))
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.image_data[image_id]["image"] = torch.from_numpy(rgb).float()


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        idx = list(self.image_data.keys())[idx]

        sample = {
            "uv": uv,
            "intrinsics": self.image_data[idx]["intrinsics"],
            "pose": self.image_data[idx]["pose"]
        }

        ground_truth = {
            "rgb": self.image_data[idx]["image"]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] =self.image_data[idx]["image"][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return self.scale_mat
