import fnmatch
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class PersonDataset(Dataset):
    def __init__(self, img_dir, labels_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(labels_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and labels don't match"
        self.len_dataset = n_images
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.imgsize = imgsize
        self.shuffle = shuffle
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.labels_paths = []
        for img_name in self.img_names:
            label_path = os.path.join(self.labels_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.labels_paths.append(label_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[index])
        lab_path = os.path.join(self.labels_dir, self.img_names[index]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, label):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                label[:, [1]] = (label[:, [1]] * w + padding) / h
                label[:, [3]] = (label[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                label[:, [2]] = (label[:, [2]] * h + padding) / w
                label[:, [4]] = (label[:, [4]] * h / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)
        return padded_img, label

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab


class CachedPersonDataset(PersonDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        data = super().__getitem__(index)
        self.cache[index] = data
        return data
