import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import save_img, to_array, process_img, download_img


def bim(model, img_tensor, eps, alpha, num_iter):
    adv_img = img_tensor.detach().clone()
    target_class = model(img_tensor).argmax().item()
    for i in range(num_iter):
        adv_img.requires_grad = True
        logit = model(adv_img)
        loss = nn.CrossEntropyLoss()(logit, torch.tensor([target_class], device=img_tensor.device))
        model.zero_grad()
        if adv_img.grad is not None:
            adv_img.grad.data.fill_(0)
        loss.backward()
        adv_img.requires_grad = False
        gradient = adv_img.grad.data
        adv_img = adv_img + alpha * gradient.sign()
        perturbation = torch.clamp(adv_img - img_tensor, min=-eps, max=eps)
        adv_img = torch.clamp(adv_img + perturbation, min=-2, max=2)
    return adv_img.detach()





def main():
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img_tensor = process_img(img)
    adv = bim(model, img_tensor, eps=0.04, alpha=0.01, num_iter=10)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv, img.size)
    plt.title(f'Class: {labels[new_class]}')
    print(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'bim.png')

if __name__ == '__main__':
    main()