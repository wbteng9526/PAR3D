# !/bin/bash
set -x

# Set distributed training parameters
export node_rank=0  # this is the first and only node
export master_addr="localhost"  # running on local machine
export master_port=29501  # can be any free port
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_cam_ae.py \
  --data_path /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video \
  --image_size 256 \
  --precision bf16 \
  --model_type ae \
  --max_epochs 3 \
  --wandb_project plucker-tokens \
  --run_name ae-bf16 \
  --batch_size 8 \
  --out_dir /wekafs/ict/wenbinte/projects/PAR3D/cam_ae \