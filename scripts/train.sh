# !/bin/bash
set -x

# Set distributed training parameters
export node_rank=0  # this is the first and only node
export master_addr="localhost"  # running on local machine
export master_port=29500  # can be any free port
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
--nnodes=1 --nproc_per_node=2 --node_rank=0 \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_3d.py \
--cloud-save-path /wekafs/ict/wenbinte/projects/PAR3D \
--data-path /wekafs/ict/wenbinte/data/MVDataset_code_vidtok \
--global-index-file /wekafs/ict/wenbinte/data/MVDataset_code_vidtok/global_index.txt \
--results-dir ./results \
--spe-token-num 15 \
--ar-token-num 16 \
--image-size 256 \
--temporal-size 4 \
--downsample-size 8 \
--gpt-model GPT-XL \
--global-batch-size 8