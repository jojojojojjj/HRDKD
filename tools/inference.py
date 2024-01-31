# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.config import Config, DictAction
from seg.utils import register_all_modules
from seg.apis import MMSegInferencer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_dir', help='image file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 将模型加载到内存中
    inferencer = MMSegInferencer(model=cfg,
                                 weights=args.checkpoint)
    # 推理
    inferencer(args.img_dir,
               out_dir='./out_dir',
               show=False)

if __name__ == '__main__':
    main()