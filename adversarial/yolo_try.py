import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision.transforms as transforms
import patch_utils


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval()

path_img = "../adversarial/for_test.jpg"
img = Image.open(path_img).convert("RGB")
path_patch = "noisy_patch/patch_e_1500_b_16_tv_2.png"
patch = Image.open(path_patch).convert("RGB")
with torch.no_grad():
    results = model(img)
results.show()
# results.save()

p_img = img.copy()
boxes = results.xyxy[0].numpy()  # [xmin, ymin, xmax, ymax, confidence, class]
target_w = int((boxes[0][2] - boxes[0][0]) - 300)
center_x = int(boxes[0][0] + target_w // 2)
target_h = int(boxes[0][3] - boxes[0][1])
center_y = int(boxes[0][1] + target_h // 2)
t_x = int((center_x - target_w // 2))
t_y = int((center_y - target_w // 2) - 30)
resized_patch = patch.resize((target_w, target_w))
p_img.paste(resized_patch, (int(boxes[0][0]), int(boxes[0][3] // 4.7)))
# p_img.show()
with torch.no_grad():
    results_1 = model(p_img)
results_1.show()
# results_1.save()

to_pil = transforms.ToPILImage()
noise = to_pil(patch_utils.generate_patch(target_w, 'cpu', "gray"))
img.paste(noise, (int(boxes[0][0]), int(boxes[0][3] // 4.7)))
with torch.no_grad():
    results_2 = model(img)
results_2.show()
# results_2.save()
