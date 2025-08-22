import torch
import torchvision.transforms as transforms
from einops import rearrange
from pathlib import Path
import os
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import CLIPImageProcessor
from PIL import Image
from utils.geometry import get_plucker_coordinates, compress_16_to_4_karcher, compress_16_to_4_geodesic
from dataset.augmentation import center_crop_arr
import random

DL3DV_SCALE = 4

def process_transform_json(filename: Path, dataset_name: str, image_size: int, selected_indices: list = None):
    with open(filename, 'r') as f:
        cam = json.load(f)

    frames = sorted(cam["frames"], key=lambda x: x["file_path"])
    if selected_indices is not None:
        frames = [f for f in frames if int(f["file_path"].split("_")[-1].split(".")[0]) in selected_indices]
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

    resize_scale = min(h / image_size, w / image_size)
    for frame in frames:
        c2w = torch.tensor(frame["transform_matrix"]).float()
        if dataset_name == "dl3dv" or dataset_name == "co3d" or dataset_name == "MVImgNet":
            c2w[2, :] *= -1
            c2w = c2w[torch.tensor([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
        intrinsic = torch.eye(3).float()
        intrinsic[0, 0] = fx / resize_scale / image_size
        intrinsic[0, 2] = cx / w
        intrinsic[1, 1] = fy / resize_scale / image_size
        intrinsic[1, 2] = cy / h
        
        extrinsics.append(c2w)
        intrinsics.append(intrinsic)

    extrinsics = torch.stack(extrinsics)
    intrinsics = torch.stack(intrinsics)
    return extrinsics, intrinsics

class MultiViewDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.root = args.data_path
        if train:
            global_index_file = os.path.join(args.data_path, "global_index_train.txt")
        else:
            global_index_file = os.path.join(args.data_path, "global_index_val.txt")
        with open(global_index_file, 'r') as f:
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
        first_image_file = Path(self.root) / data_index / "first_frame.png"

        indices = torch.from_numpy(np.load(indices_file)).long()

        extrinsics, intrinsics = process_transform_json(transform_file, dataset_name, self.image_size)
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
                [self.image_size, self.image_size]
        )
        # camera_matrix = torch.cat([extrinsics[:, :3, :], intrinsics], dim=-1)
        # viewmats = compress_16_to_4_geodesic(extrinsics_rel)
        
        clip_input = self.feature_extractor(images=Image.open(first_image_file).convert("RGB"), return_tensors="pt").pixel_values[0]

        return indices[1:], plucker_coords, clip_input


class MultiViewCamDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.root = args.data_path
        if train:
            global_index_file = os.path.join(args.data_path, "global_index_train.txt")
        else:
            global_index_file = os.path.join(args.data_path, "global_index_val.txt")
        with open(global_index_file, 'r') as f:
            self.global_index = [line.strip() for line in f.readlines()]
        
        self.image_size = args.image_size
    
    def __len__(self):
        return len(self.global_index)
    
    def __getitem__(self, idx):
        data_index = self.global_index[idx]
        dataset_name = data_index.split("/")[0]
        transform_file = Path(self.root) / data_index / "transforms.json"

        extrinsics, intrinsics = process_transform_json(transform_file, dataset_name, self.image_size)
        # get plucker coordinates (ray embeddings)
        extrinsics_src = extrinsics[0]

        plucker_coords = get_plucker_coordinates(
                extrinsics_src,
                extrinsics,
                intrinsics,
                [self.image_size, self.image_size]
        )
  
        return plucker_coords

class RE10KVideoEvalDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.evaluate_file = args.evaluate_file
        with open(self.evaluate_file, 'r') as f:
            self.data_list = json.load(f)
        
        self.index_file = args.index_file
        with open(self.index_file, 'r') as f:
            self.index_meta = json.load(f)
        
        self.scenes = list(self.data_list.keys())
        self.num_frames = args.num_frames
        self.image_size = args.image_size

        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.feature_extractor = CLIPImageProcessor.from_pretrained(args.clip_ckpt)
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        while True:
            try:
                return self.__getitem(idx)
            except Exception as e:
                # print(e)
                idx = random.randint(0, len(self.scenes) - 1)
    
    def __getitem(self, idx):
        scene_name = self.scenes[idx]
        scene_meta = self.data_list[scene_name]

        scene_folder = self.index_meta[scene_name].split(".")[0]
        
        target_indices = scene_meta["target"][:self.num_frames]
        context_indices = scene_meta["context"]

        image_dir = Path(self.data_dir) / scene_folder / scene_name / "images"

        first_frame = Image.open(image_dir / f"frame_{context_indices[0]:06d}.png").convert("RGB")
        target_frames = [Image.open(image_dir / f"frame_{target_index:06d}.png").convert("RGB") for target_index in target_indices]

        all_frames = [first_frame] + target_frames
        all_frames = [self.transform(f) for f in all_frames]
        frame_tensors = torch.stack(all_frames)

        transform_file = Path(self.data_dir) / scene_folder / scene_name / "transforms.json"
        extrinsics, intrinsics = process_transform_json(transform_file, "re10k", self.image_size, target_indices)
        
        plucker_coords = get_plucker_coordinates(
                extrinsics[0],
                extrinsics,
                intrinsics,
                [self.image_size, self.image_size]
        )
        clip_input = self.feature_extractor(images=first_frame, return_tensors="pt").pixel_values[0]

        return frame_tensors, plucker_coords, clip_input, scene_name

    
