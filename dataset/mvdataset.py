import torch
from einops import rearrange
from pathlib import Path
import os
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import CLIPImageProcessor
from PIL import Image
from utils.geometry import get_plucker_coordinates

DL3DV_SCALE = 4


class MultiViewDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.root = args.data_path
        with open(args.global_index_file, 'r') as f:
            self.global_index = [line.strip() for line in f.readlines()]
        
        self.feature_extractor = CLIPImageProcessor.from_pretrained(args.clip_ckpt)
        self.image_size = args.image_size
    
    def __len__(self):
        return len(self.global_index)
    
    def __getitem__(self, idx):
        data_index = self.global_index[idx]
        dataset_name = data_index.split("/")[0]
        indices_file = Path(self.root) / data_index / "indices.npy"
        transform_file = Path(self.root) / data_index / "transforms.json"
        first_image_file = Path(self.root) / data_index / "frist_frame.png"

        indices = torch.from_numpy(np.load(indices_file)).float()

        extrinsics, intrinsics = self.process_transform_json(transform_file, dataset_name)
        # get plucker coordinates (ray embeddings)
        extrinsics_src = extrinsics[0]
        c2w_src = torch.linalg.inv(extrinsics_src)
        # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
        extrinsics_rel = torch.einsum(
            "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
        )
        plucker_coords = get_plucker_coordinates(
                extrinsics_src,
                extrinsics,
                intrinsics,
                self.image_size
        )
        # camera_matrix = torch.cat([extrinsics[:, :3, :], intrinsics], dim=-1)
        clip_input = self.feature_extractor(image=Image.open(first_image_file).convert("RGB"), return_tensors="pt").pixel_values[0]

        return indices, plucker_coords, clip_input

    def process_transform_json(self, filename: Path, dataset_name: str):
        with open(filename, 'r') as f:
            cam = json.load(f)

        frames = sorted(cam["frames"], key=lambda x: x["file_path"])
        if "fl_x" in cam and "cx" in cam:
            fx, fy, cx, cy, h, w = \
                cam["fl_x"] / DL3DV_SCALE if dataset_name == "dl3dv" else cam["fl_x"], \
                cam["fl_y"] / DL3DV_SCALE if dataset_name == "dl3dv" else cam["fl_y"], \
                cam["cx"] / DL3DV_SCALE if dataset_name == "dl3dv" else cam["cx"], \
                cam["cy"] / DL3DV_SCALE if dataset_name == "dl3dv" else cam["cy"], \
                cam["h"] // DL3DV_SCALE if dataset_name == "dl3dv" else cam["h"], \
                cam["w"] // DL3DV_SCALE if dataset_name == "dl3dv" else cam["w"]
        else:
            fx, fy, cx, cy, h, w = \
                frames[0]["fl_x"], \
                frames[0]["fl_y"], \
                frames[0]["cx"], \
                frames[0]["cy"], \
                frames[0]["h"], \
                frames[0]["w"]
            
        extrinsics, intrinsics = [], []

        resize_scale = min(h / self.image_size, w / self.image_size)
        for frame in frames:
            c2w = torch.tensor(frame["transform_matrix"]).float()
            if dataset_name == "dl3dv" or dataset_name == "co3d" or dataset_name == "MVImgNet":
                c2w[2, :] *= -1
                c2w = c2w[torch.tensor([1, 0, 2, 3]), :]
                c2w[0:3, 1:3] *= -1
            intrinsic = torch.eye(3).float()
            intrinsic[0, 0] = fx / resize_scale / self.image_size
            intrinsic[0, 2] = cx / w
            intrinsic[1, 1] = fy / resize_scale / self.image_size
            intrinsic[1, 2] = cy / h
            
            extrinsics.append(c2w)
            intrinsics.append(intrinsic)

        extrinsics = torch.stack(extrinsics)
        intrinsics = torch.stack(intrinsics)
        return extrinsics, intrinsics 
        
    