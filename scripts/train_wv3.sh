CUDA_VISIBLE_DEVICES=0 python trainer.py \
--batch_size 16 \
--epochs 600 \
--lr 0.0006 \
--ckpt 20 \
--train_set_path PATH TO TRAIN DATASET \
--checkpoint_save_path PATH TO SAVE PATH \
--hw_range LOWER UPPER \
--task 'wv3'
# --checkpoint_path ''
# --use_pretrain