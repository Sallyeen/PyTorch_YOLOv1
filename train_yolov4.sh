python train.py \
        --cuda \
        -d coco \
        -m yolov4 \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 640 \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --multi_scale \
        --multi_scale_range 10 20 \
        --mosaic \
        --scale_loss pos \
        --multi_anchor \
        --center_sample \
        --ema
                