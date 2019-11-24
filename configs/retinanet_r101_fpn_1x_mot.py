# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.5,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'MOTDataset'
data_root = 'data/Track_Test/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.pkl',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test.pkl',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/GT/Camera1/Detection/camera1.pkl',
        img_prefix='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/GT/Camera1/Detection/images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
device_ids = range(3)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r101_fpn_1x_mot'
load_from = None
resume_from = None
workflow = [('train', 1)]

skip = 1
show = False
single = True
checkpoint = 'checkpoints/faster_crnn_r101_fpn_1x_mot_50ep_101019-a3b5c112.pth'
video_name = '/media/allysakatebrillantes/MyPassport/DATASET/Thesis/ch01_07-12_10.35-1.mp4'

tracktor = dict(
    reid_weights='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Models/siamese/train/ep25036/ResNet_iter_25036.pth',
    reid_config='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Models/siamese/train/ep25036/sacred_config.yaml',
    interpolate=False,
    write_images=False,     # compile video with=`ffmpeg -f image2 -framerate 15 -i %06d.jpg -vcodec libx264 -y movie.mp4 -vf scale=320:-1`
    output_dir = 'results/tracker',
    tracker=dict(        
        detection_thresh=0.5,
        regression_thresh=0.5,           #score threshold for keeping the track alive
        detection_nms_thresh=0.5,        #NMS threshold for detection
        regression_nms_thresh=0.6,       # NMS theshold while tracking
        motion_model=False,              # use a constant velocity assumption v_t = x_t - x_t-1
        # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
        public_detections=True,          # 0 tells the tracker to use private detections (Faster R-CNN)
        max_features_num=10,             # How much last appearance features are to keep
        do_align=False,                   # Do camera motion compensation
        warp_mode='cv2.MOTION_EUCLIDEAN',  # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        number_of_iterations=100,        # maximal number of iterations (original 50)
        termination_eps=0.00001,         # Threshold increment between two iterations (original 0.001)
        do_reid=False,                    # Use siamese network to do reid
        inactive_patience=10,            # How much timesteps dead tracks are kept and cosidered for reid
        reid_sim_threshold=2.0,          # How similar do image and old track need to be to be considered the same person
        reid_iou_threshold=0.2,         # How much IoU do track and image need to be considered for matching
        img_scale = (1333, 800)
    )
)        
