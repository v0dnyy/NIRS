from PIL import Image
import torch
import numpy as np
import requests
import torch.nn.functional as F
from torchvision.transforms import (Compose, Normalize, Resize, ToTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    pred_labels = []
    confidences = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            pred_labels.append(predicted.cpu())
            confidences.append(probabilities.max(dim=1)[0].cpu())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, pred_labels, total - correct, confidences


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
    tensor = tensor_.unsqueeze(0).to(device)
    return tensor


def read_img(path):
    img = Image.open(path).convert('RGB')
    transform = Compose([Resize((244, 244)),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])
    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0).to(device)
    return tensor, img.size


def to_array(tensor):
    tensor_ = tensor.squeeze().cpu()
    unnormalize_transform = Compose([Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                               std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                     ])
    arr_ = unnormalize_transform(tensor_)
    arr = arr_.permute(1, 2, 0).detach().numpy() * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
