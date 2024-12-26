from PIL import Image
import torch
import numpy as np
import requests
from torchvision.transforms import (Compose, Normalize, Resize,
                                    ToTensor)


def compute_gradient(func, inp, **kwargs):
    inp.requires_grad = True
    loss = func(inp, **kwargs)
    loss.backward()
    inp.requires_grad = False
    return inp.grad.data


def download_img():
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image


def save_img(arr, file_name):
    image = Image.fromarray(arr, mode='RGB')
    image.save(file_name)


def process_img(img):
    transform = Compose([Resize((244, 244)),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])
    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0)
    return tensor


def read_image(path):
    img = Image.open(path).convert('RGB')
    transform = Compose([Resize((244, 244)),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])
    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0)
    return tensor, img.size


def to_array(tensor, orig_size):
    tensor_ = tensor.squeeze()
    unnormalize_transform = Compose([Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                               std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                     Resize((orig_size[1], orig_size[0]))])
    arr_ = unnormalize_transform(tensor_)
    arr = arr_.permute(1, 2, 0).detach().numpy() * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def scale_grad(grad):
    grad_arr = torch.abs(grad).mean(dim=1).detach().permute(1, 2, 0)
    grad_arr /= grad_arr.quantile(0.98)
    grad_arr = torch.clamp(grad_arr, 0, 1)
    return grad_arr.numpy()
