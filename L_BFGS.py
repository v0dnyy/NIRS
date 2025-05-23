import torch
import torchvision.models as models
import torch.nn as nn
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import numpy as np
from my_utils import save_img, to_array, process_img, download_img


def calc_loss(adv_img_np, c, model, img_tensor, target_label, device):
    adv_img = torch.from_numpy(adv_img_np.reshape(img_tensor.size())).float().to(device)
    adv_img.requires_grad = True
    logit = model(adv_img)
    ce_loss = nn.CrossEntropyLoss()(logit, torch.tensor([target_label], device=device))
    l2 = (torch.norm(adv_img - img_tensor)) ** 2
    loss = ce_loss * c + l2
    loss.backward()
    grad = adv_img.grad.data.cpu().numpy().flatten().astype(np.float64)
    loss = loss.data.cpu().numpy().flatten().astype(np.float64)
    return loss, grad


def l_bfgs_l(model, c, adv_img_np, img_tensor, iter_num, target_label, device):
    minimum, maximum = -255, 255
    bounds = [(minimum, maximum)] * len(adv_img_np)
    approx_grad_eps = (maximum - minimum) / 100.0
    adv_img_np, f, d = fmin_l_bfgs_b(
        calc_loss,
        adv_img_np.flatten(),
        args=(c, model, img_tensor, target_label, device),
        bounds=bounds,
        m=15,
        maxiter=iter_num,
        factr=1e10,
        maxls=5,
        epsilon=approx_grad_eps
    )
    if np.amax(adv_img_np) > maximum or np.amin(adv_img_np) < minimum:
        adv_img_np = np.clip(adv_img_np, minimum, maximum)
    adv_img = torch.from_numpy(adv_img_np.reshape(img_tensor.shape)).float().to(device)
    logit = model(adv_img)
    adv_label = logit.argmax().item()
    is_adversarial = (target_label == adv_label)

    return adv_img, is_adversarial, adv_label


def l_bfgs(model, img_tensor, target_label, eps, num_iter, device):
    adv_img_np = img_tensor.detach().clone().cpu().numpy().flatten().astype(np.float64)
    c = eps
    is_adv = False
    for i in range(30):
        c = 2 * c
        adv_img, is_adv, adv_label = l_bfgs_l(model, c, adv_img_np, img_tensor, num_iter, target_label, device)
        if is_adv:
            break
    if not is_adv:
        return torch.from_numpy(adv_img_np.reshape(img_tensor.shape)).float().to(device)
    c_low = 0
    c_high = c
    while c_high - c_low > eps:
        c_half = (c_low + c_high) / 2
        is_adversary = l_bfgs_l(model, c_half, adv_img_np, img_tensor, num_iter, target_label, device)[1]
        if is_adversary:
            c_high = c_high - eps
        else:
            c_low = c_half
    adv_img, is_adv, adv_label = l_bfgs_l(model, c_high, adv_img_np, img_tensor, num_iter, target_label, device)
    return adv_img.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    target_label = 52
    print(f'Target label: {labels[target_label]}')
    img_tensor = process_img(img)
    adv = l_bfgs(model, img_tensor, target_label, 0.005, 4, device)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv)
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'l_bfgs.png')


if __name__ == '__main__':
    main()
