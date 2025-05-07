import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from my_utils import save_img, to_array, process_img, download_img


def pgd(model, img_tensor, eps, alpha, num_iter, device):
    adv_img = img_tensor.detach().clone().to(device)
    current_class = model(img_tensor).argmax().item()
    for i in range(num_iter):
        adv_img.requires_grad = True
        logit = model(adv_img)
        loss = nn.CrossEntropyLoss()(logit, torch.tensor([current_class], device=device))
        model.zero_grad()
        loss.backward()
        adv_img.requires_grad = False
        grad_sign = adv_img.grad.data.sign()
        adv_img += alpha * grad_sign
        adv_img = torch.clamp(adv_img, min=img_tensor - eps, max=img_tensor + eps)
        adv_img = torch.clamp(adv_img, min=-2, max=2)
    return adv_img.detach()

def main():
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img_tensor = process_img(img)
    adv = pgd(model, img_tensor, 0.04, 0.007,20)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv, img.size)
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'pgd.png')


if __name__ == '__main__':
    main()