# model settings
input_size = 512
# TODO missing keys in source state_dict: extra.1.weight, l2_norm.weight, extra.5.bias, 
# extra.8.bias, extra.0.bias, extra.3.weight, extra.1.bias, extra.0.weight, extra.9.bias, 
# extra.6.bias, extra.2.weight, extra.4.weight, extra.7.bias, extra.8.weight, extra.2.bias, 
# extra.4.bias, extra.7.weight, extra.5.weight, extra.6.weight, extra.9.weight, extra.3.bias
model = dict(
    type='SingleStageDetector',
    pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='SSDVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        num_classes=5,
        anchor_strides=(8, 16, 32, 64, 128, 256, 512),
        basesize_ratio_range=(0.15, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
# model training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.50),
    track=dict(
        score_thr=0.5, nms=dict(type='nms', iou_thr=0.3), max_per_img=10),
    regress=dict(
        score_thr=0.5, nms=dict(type='nms', iou_thr=0.6), max_per_img=10),
    min_bbox_size=0,
    score_thr=0.2,
    max_per_img=10)
# dataset settings
dataset_type = 'MOTDataset'
data_root = 'data/Track_Test/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train/train.pkl',
            img_prefix=data_root + 'train/images/',
            pipeline=train_pipeline)),
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
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 20])
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
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssd512_mot'
load_from = None
resume_from = None
workflow = [('train', 1)]

skip = 1
show = False
single = True
checkpoint = 'checkpoints/ssd_vgg_mot_110719-38442c1e.pth'
video_name = '/media/allysakatebrillantes/MyPassport/DATASET/Thesis/ch01_07-12_10.35-1.mp4'

tracktor = dict(
    reid_weights='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Models/siamese/train/ep25036/ResNet_iter_25036.pth',
    reid_config='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Models/siamese/train/ep25036/sacred_config.yaml',
    interpolate=False,
    write_images=False,     # compile video with=`ffmpeg -f image2 -framerate 15 -i %06d.jpg -vcodec libx264 -y movie.mp4 -vf scale=320:-1`
    output_dir = '/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Result/SSD',
    tracker=dict(        
        detection_thresh=0.4,
        regression_thresh=0.4,           # score threshold for keeping the track alive
        detection_nms_thresh=0.3,        # NMS threshold for detection
        regression_nms_thresh=0.3,       # NMS theshold while tracking
        motion_model=False,              # use a constant velocity assumption v_t = x_t - x_t-1
        # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
        public_detections=True,          # 0 tells the tracker to use private detections (Faster R-CNN)
        max_features_num=10,             # How much last appearance features are to keep
        do_align=False,                   # Do camera motion compensation
        warp_mode='cv2.MOTION_EUCLIDEAN',  # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        number_of_iterations=100,        # maximal number of iterations (original 50)
        termination_eps=0.00001,         # Threshold increment between two iterations (original 0.001)
        previous_reid=False,                 # Use siamese network to do reid
        inactive_patience=10,            # How much timesteps dead tracks are kept and cosidered for reid
        reid_sim_threshold=2.0,          # How similar do image and old track need to be to be considered the same person
        reid_iou_threshold=0.2,         # How much IoU do track and image need to be considered for matching
        img_scale = (1333, 800)
    )
)
ocr = dict(
    saved_model='/media/allysakatebrillantes/MyPassport/DATASET/Thesis/Models/attn/123456/best_accuracy.pth',
    imgH=32,
    imgW=100,
    batch_max_length=25,
    rgb = False,
    character='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ#',
    sensitive= True,
    PAD= False,
    Transformation='TPS',
    FeatureExtraction='ResNet',
    SequenceModeling='BiLSTM',
    Prediction='Attn',
    num_fiducial=20,
    input_channel=1,
    output_channel=512,
    hidden_size=256
)