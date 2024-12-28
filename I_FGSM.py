import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import save_img, download_img, to_array, process_img


def i_fgsm(model, img_tensor, eps, num_iter):
    adv_img = img_tensor.detach().clone()
    current_class = model(img_tensor).argmax().item()
    for i in range(num_iter):
        adv_img.requires_grad = True
        logit = model(adv_img)
        loss = nn.CrossEntropyLoss()(logit,  torch.tensor([current_class], device=img_tensor.device))
        model.zero_grad()
        loss.backward()
        adv_img.requires_grad = False
        adv_img = adv_img + eps * adv_img.grad.data.sign()
        adv_img = torch.clamp(adv_img, -2, 2)
    return adv_img.detach()


def main():
    img = download_img()
    img.save('orig.png')
    img_tensor = process_img(img)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    adv = i_fgsm(model, img_tensor, 0.02, 15)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv, img.size)
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'i-fgsm.png')


if __name__ == '__main__':
    main()
