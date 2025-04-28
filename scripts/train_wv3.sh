CUDA_VISIBLE_DEVICES=0 python trainer.py \
--batch_size 16 \
--epochs 600 \
--lr 0.0006 \
--ckpt 20 \
--train_set_path /home/zhaohui1.wang/github/datasets/PanCollection/train_wv3.h5 \
--checkpoint_save_path /home/zhaohui1.wang/github/ARConv/output \
--hw_range 128 1024 \
--task 'wv3'
# --checkpoint_path ''
# --use_pretrain
# bash scripts/train_wv3.sh
# --hw_range LOWER