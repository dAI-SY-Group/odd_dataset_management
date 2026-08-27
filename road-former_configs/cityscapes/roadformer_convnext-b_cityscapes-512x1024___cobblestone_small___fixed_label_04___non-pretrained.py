dataset_type = 'MMCityscapesDataset'
data_root = './cityscapes/cobblestone_small___fixed_label_04/data'
sample_scale = (1024, 512)
train_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[
            512, 614, 716, 819, 921, 1024, 1126, 1228, 1331, 1433, 1536, 1638,
            1740, 1843, 1945, 2048
        ],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(1024, 512)),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(1024, 512)),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root = './cityscapes/cobblestone_small___fixed_label_04/data',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='leftImg8bit_trainvaltest/leftImg8bit/train',
            disp_path='disparity_trainvaltest/disparity/train',
            normal_path='sne/train',
            seg_map_path='gtFine_trainvaltest/gtFine/train'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
            dict(
                type='RandomChoiceResize',
                scales=[
                    512, 614, 716, 819, 921, 1024, 1126, 1228, 1331, 1433,
                    1536, 1638, 1740, 1843, 1945, 2048
                ],
                resize_type='ResizeShortestEdge',
                max_size=4096),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root = './cityscapes/cobblestone_small___fixed_label_04/data',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='leftImg8bit_trainvaltest/leftImg8bit/val',
            disp_path='disparity_trainvaltest/disparit/val',
            normal_path='sne/val',
            seg_map_path='gtFine_trainvaltest/gtFine/val'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='Resize', scale=(1024, 512)),
            dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root = './cityscapes/cobblestone_small___fixed_label_04/data',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='leftImg8bit_trainvaltest/leftImg8bit/val',
            disp_path='disparity_trainvaltest/disparity/val',
            normal_path='sne/val',
            seg_map_path='gtFine_trainvaltest/gtFine/val'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='Resize', scale=(1024, 512)),
            dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
pretrained = None
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0, 0, 0, 0],
    std=[1, 1, 1, 1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 1024))
num_classes = 20
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0, 0, 0, 0, 0, 0],
        std=[1, 1, 1, 1, 1, 1],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024)),
    backbone=dict(
        type='mmpretrain_custom.TwinConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=None),
    decode_head=dict(
        type='RoadFormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=20,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet_custom.RoadFormerPixelDecoder',
            img_scale=(512, 1024),
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet_custom.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='mmdet_custom.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet_custom.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet_custom.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet_custom.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet_custom.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet_custom.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet_custom.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-08, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=5.0))
param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=50, by_epoch=True)
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1, draw=False))
default_scope = 'mmseg_custom'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
log_level = 'INFO'
resume = False
tta_model = dict(type='SegTTAModel')
launcher = 'none'
work_dir = './work_dirs/roadformer_convnext-b_cityscapes-512x1024___cobblestone_small___fixed_label_04___non-pretrained'
