import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation, crop_img

class TrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hazy_dir = os.path.join("/mnt/d/image_dehazing_master/data2/data/RSHaze/train", "hazy")
        self.gt_dir = os.path.join("/mnt/d/image_dehazing_master/data2/data/RSHaze/train", "GT")
        self.data_ids = [f for f in os.listdir(self.hazy_dir) if f.endswith('.png')]
        random.shuffle(self.data_ids)
        self.toTensor = ToTensor()
        print(f"Total number of training data: {len(self.data_ids)}")

    def _crop_patch(self, img1, img2):
        H, W = img1.shape[0], img1.shape[1]
        ph = self.args.patch_size
        ind_H = random.randint(0, H - ph)
        ind_W = random.randint(0, W - ph)
        patch1 = img1[ind_H:ind_H+ph, ind_W:ind_W+ph]
        patch2 = img2[ind_H:ind_H+ph, ind_W:ind_W+ph]
        return patch1, patch2

    def __getitem__(self, idx):
        img_name = self.data_ids[idx]
        hazy_path = os.path.join(self.hazy_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        hazy_img = crop_img(np.array(Image.open(hazy_path).convert('RGB')), base=16)
        gt_img = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)
        hazy_patch, gt_patch = self._crop_patch(hazy_img, gt_img)
        hazy_patch, gt_patch = random_augmentation(hazy_patch, gt_patch)
        hazy_patch = self.toTensor(hazy_patch)
        gt_patch = self.toTensor(gt_patch)
        return hazy_patch, gt_patch

    def __len__(self):
        return len(self.data_ids)


class TestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hazy_dir = os.path.join("/mnt/d/image_dehazing_master/data2/data/RSHaze/test", "hazy")
        self.gt_dir = os.path.join("/mnt/d/image_dehazing_master/data2/data/RSHaze/test", "GT")
        self.data_ids = [f for f in os.listdir(self.hazy_dir) if f.endswith('.png')]
        self.toTensor = ToTensor()
        print(f"Total number of test data: {len(self.data_ids)}")

    def __getitem__(self, idx):
        img_name = self.data_ids[idx]
        hazy_path = os.path.join(self.hazy_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        hazy_img = crop_img(np.array(Image.open(hazy_path).convert('RGB')), base=16)
        gt_img = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)
        hazy_img = self.toTensor(hazy_img)
        gt_img = self.toTensor(gt_img)
        return img_name, hazy_img, gt_img

    def __len__(self):
        return len(self.data_ids)