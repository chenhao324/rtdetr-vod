"""by lyuwenyu"""
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
def main(args, ) -> None:
    dist.init_distributed()
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/rtdetr/rtdetr_r101vd_6x_coco.yml', type=str,)
    parser.add_argument('--resume', '-r', type=str, )
    #parser.add_argument('--tuning', '-t', default='./output/rtdetr_r18vd_6x_coco/bs8_r18_box_no_fam.pth', type=str,)
    #parser.add_argument('--tuning', '-t', default='../../../../../../data0/RT-DETR/official_pretrain_checkpoints/rtdetr_r18vd_6x_coco_from_paddle.pth', type=str, )
    #parser.add_argument('--tuning', '-t', default='VID_pretrained_checkpoints/bs16_r101_obj365.pth', type=str, )
    parser.add_argument('--tuning', '-t', default='../../../../../../data0/RT-DETR/VID_pretrained_checkpoints/bs16_r101_obj365_det.pth', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    args = parser.parse_args()
    main(args)
