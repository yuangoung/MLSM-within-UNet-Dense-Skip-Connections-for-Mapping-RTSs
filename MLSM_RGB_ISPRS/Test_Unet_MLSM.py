
from __future__ import print_function, division

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from modeling.unetplusplus_mlsm import *


def get_rrhtdata_labels():
    """Load the mapping that associates rrhtdata classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([
        [0,   0,   0],   # class 0: background
        [128, 0,   0],   # class 1: RTSs
    ], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="PyTorch AmRTSNet_RGB Inference")
    parser.add_argument('--in-path', type=str, default=r'F:\RRHT\Datasets\IMG_25m')
    parser.add_argument('--out-path', type=str, default=r'F:\RRHT\output_MLSM_asks_2026')
    parser.add_argument('--ckpt', type=str,
                        default=r'F:\UESTC20250913\MLSM_RGB_ISPRS\run\rrhtdata\UNetPlusPlusMLSM-UNet++_VGG\model_best.pth.tar')
    parser.add_argument('--backbone', type=str, default='mobilenet')
    parser.add_argument('--out-stride', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='rrhtdata')
    args = parser.parse_args()

    # GPU
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{args.gpu_ids[0]}' if args.cuda else 'cpu')

    # load_state_dict checkpoint
    model = UNetPlusPlusMLSM(
        num_classes=args.num_classes,
        backbone=args.backbone,
        output_stride=args.out_stride,
        sync_bn=False,
        freeze_bn=False
    )
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    os.makedirs(args.out_path, exist_ok=True)

    # RGB normalization
    # 这里先沿用你 RTS_Segmentation 里对应的 RGB mean/std 风格
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # LOAD label colors
    label_colors = get_rrhtdata_labels()

    for filename in os.listdir(args.in_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue

        input_path = os.path.join(args.in_path, filename)
        output_path = os.path.join(
            args.out_path,
            os.path.splitext(filename)[0] + '_mask.png'
        )

        # Read RGB image exactly in the same style as RTS_Segmentation
        # output: numpy.float32, shape = (H, W, 3)
        img = Image.open(input_path).convert('RGB')
        img = np.array(img, dtype=np.float32)

        # normalization: (img - mean) / std
        img = (img - mean) / std

        # to tensor: (1, 3, H, W)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)) \
                      .unsqueeze(0) \
                      .to(device=device, dtype=torch.float32)

        # inference
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]

            out = F.interpolate(
                out,
                size=tensor.shape[2:],
                mode='bilinear',
                align_corners=True
            )

            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # label to RGB mask
        color_mask = label_colors[pred]  # (H, W, 3)
        img_pil = Image.fromarray(color_mask)
        img_pil.save(output_path)

        print(f"Processed {filename} -> {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()