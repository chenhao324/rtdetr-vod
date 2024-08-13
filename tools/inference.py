import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import glob
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
import random
import tqdm
class_real_names = ['airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']


class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None  # 保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)


class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        start1 = time.time()
        outputs = self.model(images)
        print(f"推理耗时model：{time.time() - start1:.4f}s")
        return self.postprocessor(outputs, orig_target_sizes)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtdetr/rtdetr_r101vd_6x_coco.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="../../../../../../data0/RT-DETR/VID_pretrained_checkpoints/bs16_r101_obj365_det.pth", help="权重文件路径")
    # parser.add_argument("--ckpt", default="final_checkpoints/bs8_r101_trans_all_box_cls_obj365.pth", help="权重文件路径")
    parser.add_argument("--test_data", default="test_data/fuse1", help="待推理图片路径")
    parser.add_argument("--detect_mode", default='img', help="which mode of testing: vid or img")
    parser.add_argument("--output_dir", default="test_data/00118001", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda:7")

    return parser

def main(args):
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)
    res = []

    data_path_names = []
    num = 0
    if args.detect_mode == 'img':
        if '.' not in Path(args.test_data).name:
            data_dirs = glob.glob(args.test_data + '/*')
            for data_dir in data_dirs:
                data_path_name = []
                frames = []
                data_dr = glob.glob(data_dir + '/*')
                for ddir in data_dr:
                    data_path = Path(ddir)
                    img = reader(data_path).squeeze(0).to(device)  # (1, 3, 640, 640)
                    size = torch.tensor([[img.shape[1], img.shape[2]]]).to(device)  # (640, 640)
                    frames.append(img)
                    data_path_name.append(data_path)
                frame_len = len(frames)
                index_list = list(range(frame_len))
                random.seed(43)
                random.shuffle(index_list)
                random.seed(43)
                random.shuffle(frames)
                random.seed(43)
                random.shuffle(data_path_name)

                split_num = int(frame_len / (8))  #

                for i in range(split_num):
                    res.append(frames[i * 8:(i + 1) * 8])
                res.append(frames[(i + 1) * 8:])
                data_path_names = data_path_names + data_path_name

            for ele in res:
                if ele == []: continue
                ele = torch.stack(ele)
                start2 = time.time()
                output = model(ele, size)
                print(f"推理耗时model+postprocess：{time.time() - start2:.4f}s")
                labels, boxes, scores = output
                thrh = 0.6
                for i in range(ele.shape[0]):
                    im = Image.open(data_path_names[num+i]).convert('RGB')
                    width, height = Image.open(data_path_names[num+i]).size
                    ratio_w = width/640
                    ratio_h = height/640
                    # im = Image.new("RGB", (640, 640), "white")
                    draw = ImageDraw.Draw(im)
                    scr = scores[i]
                    if len(scr[scr > thrh]) > 0:
                        max_idx = torch.argmax(scr[scr > thrh])
                        lab = labels[i][scr > thrh][max_idx].unsqueeze(0)
                        box = boxes[i][scr > thrh][max_idx].unsqueeze(0)
                    else:
                        lab = labels[i][scr > thrh]
                        box = boxes[i][scr > thrh]
                    # lab = labels[i][scr > thrh]
                    # box = boxes[i][scr > thrh]
                    box[:, 0] = box[:, 0] * ratio_w
                    box[:, 2] = box[:, 2] * ratio_w
                    box[:, 1] = box[:, 1] * ratio_h
                    box[:, 3] = box[:, 3] * ratio_h
                    for x, b in enumerate(box):
                        draw.rectangle(list(b), outline='red', width=18)
                        if len(str(class_real_names[lab[x].item()])) >= 9:
                            draw.rectangle([(b[0], b[1] - 65), (b[0] + 500, b[1])], fill='red')
                        else:
                            draw.rectangle([(b[0], b[1] - 65), (b[0] + 350, b[1])], fill='red')
                        draw.text((b[0], b[1]-65), text=str(class_real_names[lab[x].item()])+' '+str(round(scr[x].item(), 2)), fill='white', font_size=70)
                    save_path = Path(args.output_dir) / (data_path_names[num + i].parts[2] + data_path_names[num+i].name)
                    im.save(save_path)
                    print(f"检测结果已保存至:{save_path}")
                num = num + len(ele)

        elif Path(args.test_data).name.split('.')[1] in ['JPEG', 'jpg', 'png']:
            data_path = Path(args.test_data)
            img = reader(data_path).to(device)  #(1, 3, 640, 640)
            size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)  #(640, 640)
            start = time.time()
            output = model(img, size)
            print(f"推理耗时：{time.time() - start:.4f}s")
            labels, boxes, scores = output
            im = reader.pil_img
            draw = ImageDraw.Draw(im)
            thrh = 0.6

            for i in range(img.shape[0]):

                scr = scores[i]
                lab = labels[i][scr > thrh]
                box = boxes[i][scr > thrh]

                for x, b in enumerate(box):
                    draw.rectangle(list(b), outline='red', )
                    draw.text((b[0], b[1]), text=str(class_real_names[lab[x].item()])+':'+str(scr[x]), fill='blue', )

            save_path = Path(args.output_dir) / data_path.name
            im.save(save_path)
            print(f"检测结果已保存至:{save_path}")

    # elif args.detect_model == 'vid':
        # data_dirs = glob.glob(args.test_data + '/*')
        # for data_dir in data_dirs:
        #     data_path = Path(data_dir)
        #
        #     img = reader(data_path).to(device)  # (1, 3, 640, 640)
        #     size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)  # (640, 640)
        #     start = time.time()
        #     output = model(img, size)
        #     print(f"推理耗时model+postprocess：{time.time() - start:.4f}s")
        #     labels, boxes, scores = output
        #     im = reader.pil_img
        #     draw = ImageDraw.Draw(im)
        #     thrh = 0.6
        #     for i in range(img.shape[0]):
        #         scr = scores[i]
        #         lab = labels[i][scr > thrh]
        #         box = boxes[i][scr > thrh]
        #         for b in box:
        #             draw.rectangle(list(b), outline='red', )
        #             draw.text((b[0], b[1]), text=str(class_real_names[lab[i].item()]), fill='blue', )
        #     save_path = Path(args.output_dir) / data_path.name
        #     im.save(save_path)
        #     print(f"检测结果已保存至:{save_path}")

    else:
        print("mode error! please input img or vid")

if __name__ == "__main__":
    main(get_argparser().parse_args())





