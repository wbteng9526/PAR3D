import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from PIL import Image

SELECTED_DATASET = ["re10k", "acid"]

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class MultiViewVideoDataset(Dataset):
    def __init__(self, data_dir, global_index_file, num_frames=16, image_size=256, is_causal=False):
        self.data_dir = data_dir
        with open(global_index_file, 'r') as f:
            self.global_index = [line.strip() for line in f.readlines()]
        self.global_index = [f for f in self.global_index if f.split("/")[0] in SELECTED_DATASET]
        self.num_frames = num_frames
        self.is_causal = is_causal

        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.global_index)
    
    def __getitem__(self, idx):
        while True:
            try:
                example = self.__getitem(idx)
                break
            except Exception as e:
                print(f"Error: {e}")
                idx = torch.randint(0, len(self.global_index), (1,)).item()
        return example

    
    def __getitem(self, idx):
        data_index = self.global_index[idx]
        image_dir = os.path.join(self.data_dir, data_index, "images")
        image_files = sorted(os.listdir(image_dir))
        image_paths = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg") or f.endswith(".png")]

        num_total_frames = len(image_paths)
        if num_total_frames < self.num_frames:
            raise ValueError(f"Video {data_index} has less than {self.num_frames} frames")
        
        max_stride = max(1, num_total_frames // self.num_frames)
        stride = random.randint(1, max_stride)

        frame_indices = list(range(0, num_total_frames, stride))
        if len(frame_indices) < self.num_frames:
            frame_indices = frame_indices + [frame_indices[-1]] * (self.num_frames - len(frame_indices))
        else:
            frame_indices = frame_indices[:self.num_frames]

        frames = [Image.open(image_paths[i]).convert("RGB") for i in frame_indices]
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames, dim=1)

        if self.is_causal:
            frames = torch.cat([frames[:, 0:1], frames], dim=1)

        sample = {}
        sample['frames'] = frames
        sample['data_index'] = data_index

        
        sample['selected_frames'] = torch.tensor(frame_indices).long()

        return sample


class MultiViewVideoIterableDataset(IterableDataset):
    def __init__(self, data_dir, global_index_file, num_frames=16, image_size=256, is_causal=False):
        self.data_dir = data_dir
        with open(global_index_file, 'r') as f:
            self.global_index = [line.strip() for line in f.readlines()]
    
        self.global_index = [f for f in self.global_index if f.split("/")[0] in SELECTED_DATASET]
        self.num_frames = num_frames
        self.is_causal = is_causal

        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    def __iter__(self):
        
        for idx in self.global_index:
            try:
            
                image_dir = os.path.join(self.data_dir, idx, "images")
                image_files = sorted(os.listdir(image_dir))
                image_paths = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg") or f.endswith(".png")]

                num_total_frames = len(image_paths)
                if num_total_frames < self.num_frames:
                    raise ValueError(f"Video {idx} has less than {self.num_frames} frames")
                
                max_stride = max(1, num_total_frames // self.num_frames)
                max_stride = min(max_stride, 5)
                if max_stride <= 2:
                    stride_choices = [random.randint(1, max_stride)]
                else:
                    stride_choices = random.sample(range(1, max_stride + 1), max_stride // 2)

                max_start_frame_choices = [max(num_total_frames - self.num_frames * stride, 0) for stride in stride_choices]

                start_frames = []
                for max_start_frame in max_start_frame_choices:
                    if max_start_frame <= 3:
                        start_frames.append([random.randint(0, max_start_frame)])
                    else:
                        start_frames.append(random.sample(range(0, max_start_frame + 1), 3))

                for stride, start_frame in zip(stride_choices, start_frames):
                    for f in start_frame:
                        frame_indices = list(range(f, num_total_frames, stride))
                        if len(frame_indices) < self.num_frames:
                            frame_indices = frame_indices + [frame_indices[-1]] * (self.num_frames - len(frame_indices))
                        else:
                            frame_indices = frame_indices[:self.num_frames]
                    

                        frames = [Image.open(image_paths[i]).convert("RGB") for i in frame_indices]
                        frames = [self.transform(frame) for frame in frames]

                        frames = torch.stack(frames, dim=1)

                        if self.is_causal:
                            frames = torch.cat([frames[:, 0:1], frames], dim=1)

                        sample = {}
                        sample['frames'] = frames
                        sample['data_index'] = idx

                        
                        sample['selected_frames'] = torch.tensor(frame_indices).long()
                        sample['stride'] = torch.tensor(stride).long()
                        sample['start_frame'] = torch.tensor(f).long()
                        sample['first_frame'] = frames[:, 0]

                        yield sample
            except Exception as e:
                print(f"Error: {e}")
                continue
        

        
