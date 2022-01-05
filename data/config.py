# config.py
import os.path


# new yolo config
'''train_cfg = {
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': [416, 416]'''

yolo_cfg = {
    # anchor size
    'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]],
}
