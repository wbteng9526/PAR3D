# !/bin/bash
set -x

# Set distributed training parameters
export node_rank=0  # this is the first and only node
export master_addr="localhost"  # running on local machine
export master_port=29500  # can be any free port
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_3d.py \
--cloud-save-path /wekafs/ict/wenbinte/projects/PAR3D/GPT-L \
--data-path /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_video \
--global-index-file /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_video/global_index.txt \
--results-dir /wekafs/ict/wenbinte/projects/PAR3D/logs \
--no-local-save \
--spe-token-num 63 \
--ar-token-num 64 \
--image-size 256 \
--temporal-size 4 \
--downsample-size 8 \
--gpt-model GPT-L \
--global-batch-size 8 \
--wandb-project PAR3D_training