import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image
import random


try:
    from .transform import *
except ImportError:
    def Compose(ops):
        return lambda img, mask: (img, mask)
CLASSES = ('Background', 'Building')
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),

        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def train_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    return aug['image'], aug['mask']


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class WHUBuildingDataset(Dataset):
    def __init__(self, data_root='data/WHU_Building/train', mode='train',
                 img_dir='image', mask_dir='label',
                 img_suffix='.tif', mask_suffix='.tif',
                 transform=train_aug, mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE):

        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if self.mode == 'val' or self.mode == 'test' or p_ratio > self.mosaic_ratio:
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:

            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask).long()

        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        dir_path = osp.join(data_root, img_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Image directory {dir_path} does not exist.")
            return []

        img_filename_list = [f for f in os.listdir(dir_path) if f.endswith(self.img_suffix)]
        img_ids = [os.path.splitext(f)[0] for f in img_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = Image.open(img_path).convert('RGB')

        mask = Image.open(mask_path).convert('L')

        mask_np = np.array(mask)


        if mask_np.max() > 1:
            mask_np = np.where(mask_np > 127, 1, 0).astype(np.uint8)
        else:
            mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8)

        mask = Image.fromarray(mask_np)

        return img, mask

    def load_mosaic_img_and_mask(self, index):

        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]

        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        croped_a = albu.RandomCrop(height=crop_size_a[1], width=crop_size_a[0])(image=img_a, mask=mask_a)
        croped_b = albu.RandomCrop(height=crop_size_b[1], width=crop_size_b[0])(image=img_b, mask=mask_b)
        croped_c = albu.RandomCrop(height=crop_size_c[1], width=crop_size_c[0])(image=img_c, mask=mask_c)
        croped_d = albu.RandomCrop(height=crop_size_d[1], width=crop_size_d[0])(image=img_d, mask=mask_d)

        top = np.concatenate((croped_a['image'], croped_b['image']), axis=1)
        bottom = np.concatenate((croped_c['image'], croped_d['image']), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((croped_a['mask'], croped_b['mask']), axis=1)
        bottom_mask = np.concatenate((croped_c['mask'], croped_d['mask']), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask

def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    patches = [mpatches.Patch(color=np.array(PALETTE[i]) / 255., label=CLASSES[i]) for i in range(len(CLASSES))]

    # img: [3, H, W] -> [H, W, 3] -> uint8
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    img = img.astype(np.uint8)

    # mask: [H, W] -> color map
    if torch.is_tensor(mask):
        mask = mask.numpy()

    mask = mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask).convert('P')

    flat_palette = np.array(PALETTE, dtype=np.uint8).flatten()
    if len(flat_palette) < 768:
        flat_palette = np.pad(flat_palette, (0, 768 - len(flat_palette)), 'constant')

    mask_pil.putpalette(flat_palette)
    mask_vis = np.array(mask_pil.convert('RGB'))

    ax1.imshow(img)
    ax1.set_title(f'Image: {img_id}')
    ax1.axis('off')

    ax2.imshow(mask_vis)
    ax2.set_title(f'Label: {img_id}\n(0:Black, 1:White)')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    DATA_ROOT = 'data/WHU_Building'

    dataset = WHUBuildingDataset(
        data_root=DATA_ROOT,
        mode='train',
        transform=train_aug,
        mosaic_ratio=0.5
    )

    print(f"Dataset Length: {len(dataset)}")

    if len(dataset) > 0:
        data = dataset[0]
        img = data['img']
        mask = data['gt_semantic_seg']
        img_id = data['img_id']

        print("\n--- Tensor Check ---")
        print(f"Image Shape: {img.shape} (Expect [3, 512, 512])")
        print(f"Mask Shape:  {mask.shape} (Expect [512, 512])")
        print(f"Mask Type:   {mask.dtype} (Expect torch.int64 or Long)")
        print(f"Unique Vals: {torch.unique(mask)} (Expect [0, 1])")

        show_mask(img, mask, img_id)