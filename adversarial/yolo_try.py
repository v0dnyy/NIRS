import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)

model.eval()

path = "/Users/vodnyy/PycharmProjects/NIRS/adversarial/dataset/test/images/crop001088_jpg.rf.5cf379ddcefb0f5055f40f75ef9245dd.jpg"
img = Image.open(path).convert("RGB")
img_resized = F.resize(img, (640, 640))  # Изменение размера
img_tensor = F.to_tensor(img_resized).unsqueeze(0)  # (1, 3, 640, 640)

# Конвертация в tensor и добавление batch dimension
# img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
# print(model)
# print("----------------------------------")
# print(model.model)

# Получение сырых выходов
with torch.no_grad():
    raw_outputs = model(img_tensor)  # теперь работает правильно


print(f"Output shapes: {[o.shape for o in raw_outputs[1]]}")
