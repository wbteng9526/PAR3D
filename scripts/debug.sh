# !/bin/bash
set -x

# Set distributed training parameters
export node_rank=0  # this is the first and only node
export master_addr="localhost"  # running on local machine
export master_port=29500  # can be any free port
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_3d.py \
--cloud-save-path /wekafs/ict/wenbinte/projects/PAR3D/GPT-XL-no-spe \
--data-path /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video \
--global-index-file /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video/global_index.txt \
--results-dir /wekafs/ict/wenbinte/projects/PAR3D/logs \
--no-local-save \
--spe-token-num 63 \
--ar-token-num 64 \
--image-size 256 \
--temporal-size 4 \
--downsample-size 8 \
--gpt-model GPT-XL \
--global-batch-size 8 \
--wandb-project PAR3D_debug \
--vocab-size 32768 \
--lr-scheduler-type cosine \
--num-warmup-steps 10000 \
--min-lr-ratio 0.05 \
--num-cycles 0.5 \
--lr 5e-4 \
--beta2 0.95 \
--no-compile \
--camera-model-path /wekafs/ict/wenbinte/projects/PAR3D/cam_ae/ckpt_best.pt \
--init-counts-path /wekafs/ict/wenbinte/projects/PAR3D/init_counts/rvq32x32x32.pt
