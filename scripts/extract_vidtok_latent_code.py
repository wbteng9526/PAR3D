import torch
import torchvision
import argparse
import os, json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from PIL import Image

from tokenizer.vidtok.data.mvdataset import MultiViewVideoDataset, MultiViewVideoIterableDataset
from tokenizer.vidtok.scripts.inference_evaluate import load_model_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vidtok_fsq_causal_488_262144.yaml")
    parser.add_argument("--ckpt", type=str, default="/wekafs/ict/wenbinte/projects/RandAR3D/tokenizer/VidTok/vidtok_fsq_causal_488_262144.ckpt")
    parser.add_argument("--data_dir", type=str, default="/wekafs/ict/wenbinte/data/MVDataset")
    parser.add_argument("--global_index_file", type=str, default="/wekafs/ict/wenbinte/projects/RandAR3D/dataset_index/global_index.txt")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="/wekafs/ict/wenbinte/data/MVDataset_vidtok_code_video")
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model_from_config(args.config, args.ckpt)
    model.to("cuda").eval()

    dataset = MultiViewVideoIterableDataset(args.data_dir, args.global_index_file, args.num_frames, args.image_size, args.is_causal)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch in tqdm(dataloader):
            frames = batch['frames'].to("cuda")
            data_index = batch['data_index']
            selected_frames = batch['selected_frames']

            # check if the output directory exists
            # check to the subdirectory re10k/train/000000
            # parts = data_index[0].split("/")
            # cur_output_dir = os.path.join(args.output_dir, parts[0], parts[1], parts[2])
            # if os.path.exists(cur_output_dir):
            #     print(f"Output directory {cur_output_dir} already exists, skipping...")
            #     continue

            # print(selected_frames)
            z, reg_log = model.encode(frames, return_reg_log=True)
            indices = reg_log['indices']

            for i in range(args.batch_size):
                cur_indice = indices[i]
                cur_data_index = data_index[i]
                cur_selected_frames = selected_frames[i]
                cur_stride = batch['stride'][i]
                cur_start_frame = batch['start_frame'][i]
                cur_first_frame = batch['first_frame'][i]

                cur_output_dir = os.path.join(args.output_dir, cur_data_index + f"_stride_{cur_stride.item()}_start_{cur_start_frame.item()}")
                os.makedirs(cur_output_dir, exist_ok=True)

                transforms_file = os.path.join(args.data_dir, cur_data_index, "transforms.json")
                with open(transforms_file, 'r') as f:
                    transforms_data = json.load(f)
                
                meta_frames = transforms_data["frames"]
                meta_frames = sorted(meta_frames, key=lambda x: x["file_path"])
                # print(len(meta_frames), cur_selected_frames)
                meta_frames_selected = [meta_frames[j.item()] for j in cur_selected_frames]
                cur_meta = {"frames": meta_frames_selected}
                with open(os.path.join(cur_output_dir, "transforms.json"), "w") as f:
                    json.dump(cur_meta, f, indent=4)
                np.save(os.path.join(cur_output_dir, "indices.npy"), cur_indice.cpu().numpy())

                cur_first_frame = ((cur_first_frame * 0.5 + 0.5) * 255).cpu().to(torch.uint8)
                Image.fromarray(cur_first_frame.permute(1, 2, 0).numpy()).save(os.path.join(cur_output_dir, "first_frame.png"))

                if args.debug:
                    x_recon = model.decode(cur_indice.unsqueeze(0), decode_from_indices=True)

                    if args.is_causal:
                        frames = frames[i:(i+1), :, 1:]
                        x_recon = x_recon[:, :, 1:]

                    in_out_combine = torch.clamp(torch.cat([frames, x_recon], dim=-1) * 0.5 + 0.5, 0, 1)
                    in_out_combine = (in_out_combine * 255).cpu().to(torch.uint8)

                    output_path = os.path.join(cur_output_dir, "recon.mp4")
                    torchvision.io.write_video(output_path, rearrange(in_out_combine[0], "c t h w -> t h w c"), fps=8, video_codec='libx264', options={'crf': '20'})



if __name__ == "__main__":
    main()


            

        
        