python train.py \
        --cuda \
        -d coco \
        -m yolo_tiny \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 640 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        --multi_scale \
        --multi_scale_range 10 16 \
        --mosaic \
        --scale_loss pos \
        --multi_anchor \
        --center_sample \
        --ema
                        