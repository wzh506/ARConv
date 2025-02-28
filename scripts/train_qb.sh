CUDA_VISIBLE_DEVICES=0 python trainer.py \
--batch_size 16 \
--epochs 200 \
--lr 0.0006 \
--ckpt 20 \
--train_set_path 'G:/PythonProjectGpu/ARConv/training_data/train_wv3.h5' \
--checkpoint_save_path 'G:/PythonProjectGpu/ARConv/checkpoints/wv3' \
--hw_range '[1, 9]' \
--task 'qb'
# --checkpoint_path ''
# --use_pretrain