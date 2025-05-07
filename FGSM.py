import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from my_utils import save_img, to_array, process_img, download_img


def fgsm(model, img_tensor, eps):
    adv_img = img_tensor.detach().clone()
    current_class = model(img_tensor).argmax().item()
    adv_img.requires_grad = True
    logit = model(adv_img)
    loss = nn.CrossEntropyLoss()(logit, torch.tensor([current_class], device=img_tensor.device))
    model.zero_grad()
    if adv_img.grad is not None:
        adv_img.grad.data.fill_(0)
    loss.backward()
    adv_img.requires_grad = False
    x_adv = adv_img + eps * adv_img.grad.data.sign()
    x_adv = torch.clamp(x_adv, -2, 2)
    return x_adv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img_tensor = process_img(img).to(device)
    adv = fgsm(model, img_tensor, 0.02)
    new_class = model(adv).argmax().item()
    # new_img = to_array(adv.cpu(), img.size)
    new_img = to_array(adv.cpu())
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'fgsm.png')


if __name__ == '__main__':
    main()
