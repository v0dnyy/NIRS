import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from my_utils import save_img, to_array, process_img, download_img


def compute_jacobian(model, image):
    var_image = image.clone().detach()
    var_image.requires_grad = True
    output = model(var_image)
    num_features = int(np.prod(var_image.shape[1:]))
    jacobian = torch.zeros([output.shape[1], num_features])

    for i in range(output.shape[1]):
        if var_image.grad is not None:
            var_image.grad.zero_()
        output[0][i].backward(retain_graph=True)
        jacobian[i] = (var_image.grad.squeeze().view(-1, num_features).clone())
    return jacobian


@torch.no_grad()
def saliency_map(jacobian, target_label, increasing, search_space, features_num):
    domain = torch.eq(search_space, 1).float()
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_label]
    others_grad = all_sum - target_grad
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float()
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
    increase_coef = increase_coef.view(-1, features_num)
    target_tmp = target_grad.clone().unsqueeze(0)
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
    alpha = target_tmp.view(-1, 1, features_num) + target_tmp.view(-1, features_num, 1)
    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, features_num) + others_tmp.view(-1, features_num, 1)

    tmp = np.ones((features_num, features_num), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte()

    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)

    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    maps = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    max_idx = torch.argmax(maps.view(-1, features_num * features_num), dim=1)
    p = torch.div(max_idx, features_num, rounding_mode="floor")
    q = max_idx - p * features_num
    return p, q


def jsma(model, img_tensor, target_label, theta, gamma):
    adv_img = img_tensor.detach().clone()
    output = model(adv_img)
    current_label = output.argmax().item()
    if theta > 0:
        increasing = True
    else:
        increasing = False
    num_features = int(np.prod(adv_img.shape[1:]))
    t_shape = adv_img.shape
    iters_num = int(np.ceil(num_features * gamma / 2.0))
    if increasing:
        search_domain = torch.lt(adv_img, 0.99)
    else:
        search_domain = torch.gt(adv_img, 0.01)
    search_domain = search_domain.view(num_features)
    i = 0
    while (i < iters_num) and (target_label != current_label) and (search_domain.sum() != 0):
        jacobian = compute_jacobian(model, adv_img)
        p1, p2 = saliency_map(jacobian, target_label, increasing, search_domain, num_features)
        flatten_sample = adv_img.view(-1, num_features)
        flatten_sample[0, p1] += theta
        flatten_sample[0, p2] += theta
        adv_img = torch.clamp(flatten_sample, min=-2.0, max=2.0)
        adv_img = adv_img.view(t_shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        output = model(adv_img)
        current_label = output.argmax().item()
        i += 1
    return adv_img


def main():
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img_tensor = process_img(img)
    adv = jsma(model, img_tensor, 50, 1.0, 0.1)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv, img.size)
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'jsma.png')


if __name__ == '__main__':
    main()
