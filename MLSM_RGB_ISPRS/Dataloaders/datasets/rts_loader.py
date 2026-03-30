from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from Dataloaders.DATApath import Path
from torchvision import transforms
from Dataloaders import custom_transforms as tr
from PIL import Image


class RTS_Segmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, base_dir=Path.db_root_dir('rrhtdata'), split='train'):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'IMG_25m')
        self._cat_dir = os.path.join(self._base_dir, 'GroundTruth_25m_8bite')
        self.split = [split] if isinstance(split, str) else sorted(split)
        self.args = args

        splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            list_path = os.path.join(splits_dir, splt + '.txt')
            assert os.path.isfile(list_path), f"划分文件不存在: {list_path}"

            with open(list_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()

            for line in lines:
                img_path = os.path.join(self._image_dir, line + '.png')
                cat_path = os.path.join(self._cat_dir, line + '.png')

                assert os.path.isfile(img_path), f"图像文件不存在: {img_path}"
                assert os.path.isfile(cat_path), f"标签文件不存在: {cat_path}"

                self.im_ids.append(line)
                self.images.append(img_path)
                self.categories.append(cat_path)

        assert len(self.images) == len(self.categories), "图像与标签数量不一致"
        print(f'Number of images in {split}: {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self._make_img_gt_point_pair(index)
        sample = {'image': image, 'label': label}

        if 'train' in self.split:
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # ---------------------------
        # 读取 RGB 三通道 PNG 影像
        # 输出: numpy.float32, shape = (H, W, 3)
        # ---------------------------
        img = Image.open(self.images[index]).convert('RGB')
        img_array = np.array(img, dtype=np.float32)

        # ---------------------------
        # 读取标签 PNG
        # 转为单通道，并统一映射到 0/1
        # 输出仍转回 PIL.Image，便于 custom_transforms 继续处理
        # ---------------------------
        label = Image.open(self.categories[index]).convert('L')
        label_array = np.array(label, dtype=np.uint8)

        # 若标签是 0/255，则转为 0/1
        # 若本身已是 0/1，也不会受影响
        label_array = (label_array > 0).astype(np.uint8)

        label = Image.fromarray(label_array)

        return img_array, label

    def transform_tr(self, sample):
        """
        训练阶段的数据增强：
        随机翻转、随机缩放裁剪、高斯模糊、归一化、ToTensor
        注意：mean/std 这里采用 RGB 8-bit 图像的常用写法。
        若你后面希望更严谨，可以再用训练集重新统计。
        """
        composed = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(
                base_size=self.args.base_size,
                crop_size=self.args.crop_size
            ),
            tr.RandomGaussianBlur(),
            tr.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375)
            ),
            tr.ToTensor()
        ])
        return composed(sample)

    def transform_val(self, sample):
        """
        验证/测试阶段的预处理：
        固定尺度裁剪、归一化、ToTensor
        """
        composed = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375)
            ),
            tr.ToTensor()
        ])
        return composed(sample)