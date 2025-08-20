_base_ = [
    '../_base_/models/hrnet.py',
    '../_base_/datasets/coco.py',
    '../_base_/schedules/schedule_210e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='TopDown',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_blocks=(4, ),
                block_inplanes=(64, ),
                block_outplanes=(64, ),
                block_modules=('BOTTLENECK', ),
                block_strides=(1, ),
                block_dilations=(1, ),
                num_branches=1,
                num_blocks_per_branch=(4, ),
                num_channels=(64, ),
                fuse_method='SUM',
                multiscale_output=True),
            stage2=dict(
                num_modules=1,
                num_blocks=(4, ),
                block_inplanes=(64, ),
                block_outplanes=(64, ),
                block_modules=('BOTTLENECK', ),
                block_strides=(1, ),
                block_dilations=(1, ),
                num_branches=1,
                num_blocks_per_branch=(4, ),
                num_channels=(64, ),
                fuse_method='SUM',
                multiscale_output=True),
            stage3=dict(
                num_modules=1,
                num_blocks=(4, ),
                block_inplanes=(64, ),
                block_outplanes=(64, ),
                block_modules=('BOTTLENECK', ),
                block_strides=(1, ),
                block_dilations=(1, ),
                num_branches=1,
                num_blocks_per_branch=(4, ),
                num_channels=(64, ),
                fuse_method='SUM',
                multiscale_output=True),
            stage4=dict(
                num_modules=1,
                num_blocks=(4, ),
                block_inplanes=(64, ),
                block_outplanes=(64, ),
                block_modules=('BOTTLENECK', ),
                block_strides=(1, ),
                block_dilations=(1, ),
                num_branches=1,
                num_blocks_per_branch=(4, ),
                num_channels=(64, ),
                fuse_method='SUM',
                multiscale_output=True)),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32'))
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=64,
        out_channels=17,
        loss_keypoint=dict(
            type='JointsMSELoss', use_target_weight=True, loss_weight=1.0),
        loss_heatmap=dict(type='CrossEntropyLoss', use_target_weight=True, loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        flip_test=False,
        post_process=True,
        shift_heatmap=True,
        modulate_kernel=11,
        use_udp=False,
        valid_score_thr=0.0,
        min_keypoints=1)
)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}train2017/',
        data_cfg=dict(
            image_size=[256, 192],
            heatmap_size=[64, 48],
            num_output_channels=17,
            num_joints=17,
            dataset_channel=['coco'],
            inference_channel=['coco']),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownRandomFlip', flip_prob=0.5),
            dict(type='TopDownRandomRotation', max_rotate_degree=30),
            dict(type='TopDownRandomZoomInOut', min_scale=0.8, max_scale=1.2),
            dict(type='TopDownRandomCrop', crop_size=(256, 192)),
            dict(type='TopDownResize', size=(256, 192)),
            dict(type='TopDownNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2),
            dict(type='Collect', keys=['img', 'target', 'target_weight'], meta_keys=['image_file'])
        ]),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}val2017/',
        data_cfg=dict(
            image_size=[256, 192],
            heatmap_size=[64, 48],
            num_output_channels=17,
            num_joints=17,
            dataset_channel=['coco'],
            inference_channel=['coco']),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownResize', size=(256, 192)),
            dict(type='TopDownNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2),
            dict(type='Collect', keys=['img'], meta_keys=['image_file'])
        ]),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}val2017/',
        data_cfg=dict(
            image_size=[256, 192],
            heatmap_size=[64, 48],
            num_output_channels=17,
            num_joints=17,
            dataset_channel=['coco'],
            inference_channel=['coco']),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownResize', size=(256, 192)),
            dict(type='TopDownNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2),
            dict(type='Collect', keys=['img'], meta_keys=['image_file'])
        ])
)

evaluation = dict(interval=1, metric='mAP')
