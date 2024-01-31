# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from mmseg.models.utils import resize
import mmcv
import torch
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.registry import VISUALIZERS
#from seg.recorder import MyRecorderManager
from seg.recorder import RecorderManager
from mmrazor.models.task_modules import ModuleOutputsRecorder, ModuleInputsRecorder, MethodOutputsRecorder
from mmrazor.visualization.local_visualizer import modify
from seg.visualization.local_visualizer import SegLocalVisualizer
from seg.apis import init_model_kd, inference_model_kd
import matplotlib.pyplot as plt
from mmseg.datasets.cityscapes import CityscapesDataset
#from seg.datasets.synapse import SynapseDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Feature map visualization')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument('--repo', help='the corresponding repo name')
    parser.add_argument(
        '--use-norm',
        action='store_true',
        help='normalize the featmap before visualization')
    parser.add_argument(
        '--overlaid', action='store_true', help='overlaid image')
    parser.add_argument(
        '--channel-reduction',
        help='Reduce multiple channels to a single channel. The optional value'
             ' is \'squeeze_mean\', \'select_max\' or \'pixel_wise_max\'.',
        default=None)
    parser.add_argument(
        '--topk',
        type=int,
        help='If channel_reduction is not None and topk > 0, it will select '
             'topk channel to show by the sum of each channel. If topk <= 0, '
             'tensor_chw is assert to be one or three.',
        default=20)
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        help='the arrangement of featmap when channel_reduction is not None '
             'and topk > 0.',
        default=[4, 5])
    parser.add_argument(
        '--resize-shape',
        nargs='+',
        type=int,
        help='the shape to scale the feature map',
        default=None)
    parser.add_argument(
        '--alpha', help='the transparency of featmap', default=0.5)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.',
        default={})

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / (std + 1e-6)
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

def main(args):
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    kd=True
    if kd:
        new_state_dict = dict()
        new_meta = checkpoint['meta']
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('architecture.'):
                new_key = key.replace('architecture.', '')
                new_state_dict[new_key] = value
    
        checkpoint = dict()
        checkpoint['meta'] = new_meta
        checkpoint['state_dict'] = new_state_dict
    model = init_model_kd(args.config, checkpoint, device=args.device)

    recorders = dict()
    recorders['preds']=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')
    # recorders['conf'] = dict(
    #     type=MethodOutputsRecorder, 
    #     source='seg.models.decode_heads.decode_head_edl.LossDualHead.get_conf')
    # recorders['logits'] = dict(
    #     type=MethodOutputsRecorder, 
    #     source='seg.models.decode_heads.decode_head.BaseDecodeHead.predict_by_feat')
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.draw_featmap = modify
    visualizer.set_dataset_meta(classes=CityscapesDataset.METAINFO['classes'],
                            palette=CityscapesDataset.METAINFO['palette'],
                            dataset_name='CityscapesDataset')
    #visualizer.set_dataset_meta(classes=SynapseDataset.METAINFO['classes'],
     #                       palette=SynapseDataset.METAINFO['palette'],
      #                      dataset_name='SynapseDataset')

    recorder_manager = RecorderManager(recorders)
    recorder_manager.initialize(model)

    with recorder_manager:
        # test a single image
        result = inference_model_kd(model, args.img, args.img.replace('img_dir', 'ann_dir').replace('jpg', 'png'))
    #print(result.pred_sem_seg.data.shape)
    #print(result.gt_sem_seg.data.shape)
    #print(result)
    #print(result.pred_sem_seg.data.shape)
    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None
    pred_sem_seg=result.pred_sem_seg
    classes=CityscapesDataset.METAINFO['classes']
    palette=CityscapesDataset.METAINFO['palette']
    num_classes = len(classes)
    pred_sem_seg = pred_sem_seg.cpu().data
    ids = np.unique(pred_sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    pred = np.zeros_like(overlaid_image, dtype=np.uint8)
    for label, color in zip(labels, colors):
        pred[pred_sem_seg[0] == label, :] = color
    mmcv.imwrite(mmcv.rgb2bgr(pred),
                        f'./out_dir/vis_cityscapes/{osp.splitext(osp.basename(args.config))[0]}/frankfurt_000_294_cirkd.jpg')

if __name__ == '__main__':
    args = parse_args()
    main(args)
