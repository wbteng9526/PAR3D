# !/bin/bash
set -x

export CUDA_VISIBLEDEVICES=2 

python tools/get_init_counts.py \
    --data_path /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video \
    --global_index_file /wekafs/ict/wenbinte/data/MVDataset_vidtok_code_fsq_causal_488_32768_video/global_index.txt \
    --save_prefix /wekafs/ict/wenbinte/projects/PAR3D/init_counts/rvq32x32x32 \
    --batch_size 128 \
    --num_workers 8 \
    --max_batches 200 \
    --k_list 32 32 32