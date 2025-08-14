# !/bin/bash
set -x

# Set distributed training parameters
export node_rank=0  # this is the first and only node
export master_addr="localhost"  # running on local machine
export master_port=29501  # can be any free port
export CUDA_VISIBLE_DEVICES=0

torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/sample/sample_3d_ddp.py \
--data-dir /wekafs/ict/wenbinte/data/MVDataset/re10k/test \
--evaluate-file ./assets/evaluation_index_re10k_video.json \
--index-file /wekafs/ict/wenbinte/data/MVDataset/re10k/test/index.json \
--tokenizer-config configs/tokenizer/vidtok_fsq_causal_488_32768.yaml \
--tokenizer-ckpt /wekafs/ict/wenbinte/projects/RandAR3D/tokenizer/VidTok/vidtok_fsq_causal_488_32768.ckpt \
--temporal-size 4 \
--num-frames 16 \
--gpt-model GPT-1B \
--gpt-ckpt /wekafs/ict/wenbinte/projects/PAR3D/GPT-1B/0095000.pt \
--sample-dir /wekafs/ict/wenbinte/projects/PAR3D/samples \
--spe-token-num 63 \
--ar-token-num 64 \
--image-size 256 \
--downsample-size 8 \
--per-proc-batch-size 4 \
--vocab-size 32768 \
--fps 8