import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import save_img, to_array, process_img, download_img


def deepfool(model, img_tensor, eps, num_iter, device):
    adv_img = img_tensor.detach().clone().to(device)
    adv_img.requires_grad = True
    logit_orig = model(img_tensor)
    logit_adv = model(adv_img)
    num_classes = logit_orig.size(1)
    i_classes = logit_orig.data[0].argsort().cpu().numpy()[::-1]
    i_classes = i_classes[0:num_classes]
    label = i_classes[0]
    adv_label = label
    # w = np.zeros(img_tensor.shape).astype(np.float32)
    # r_total = np.zeros(img_tensor.shape).astype(np.float32)
    w = torch.zeros_like(img_tensor).to(device)
    r_total = torch.zeros_like(img_tensor).to(device)

    with torch.set_grad_enabled(True):
        for i in range(num_iter):
            if adv_label != label:
                break

            perturb = 0
            logit_adv[0, i_classes[0]].backward(retain_graph=True)
            # grad_orig = adv_img.grad.data.cpu().numpy().copy()
            grad_orig = adv_img.grad.data.clone()

            for k in range(1, num_classes):
                if adv_img.grad is not None:
                    adv_img.grad.zero_()
                logit_adv[0, i_classes[k]].backward(retain_graph=True)
                # current_grad = adv_img.grad.data.cpu().numpy().copy()
                current_grad = adv_img.grad.data.clone()
                w_k = current_grad - grad_orig
                f_k = (logit_adv[0, i_classes[k]] - logit_adv[0, i_classes[0]]).data.cpu().numpy()
                # pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                pert_k = abs(f_k) / w_k.flatten().norm()
                if pert_k > perturb:
                    perturb = pert_k
                    w = w_k

            # r_i = (perturb + 1e-4) * w / np.linalg.norm(w)
            r_i = (perturb + 1e-4) * w / w.norm()
            r_total += r_i

            # adv_img = img_tensor + (1 + eps) * torch.from_numpy(r_total).to(device)
            adv_img = img_tensor + (1 + eps) * r_total
            adv_img.requires_grad = True

            logit_adv = model(adv_img)
            # adv_label = np.argmax(logit_adv.data.cpu().numpy().flatten())
            adv_label = torch.argmax(logit_adv.data.flatten()).item()

    return adv_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = download_img()
    img.save('orig.png')
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    img_tensor = process_img(img).to(device)
    adv = deepfool(model, img_tensor, 1, 10, device)
    new_class = model(adv).argmax().item()
    new_img = to_array(adv.cpu())
    plt.title(f'Class: {labels[new_class]}')
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    save_img(new_img, 'deepfool.png')


if __name__ == '__main__':
    main()
