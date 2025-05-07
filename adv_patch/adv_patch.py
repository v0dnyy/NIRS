import torch
from torchvision import models
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

from p_utils import init_patch, mask_generation, test_patch
from load_ds import load_train_teat_data
from my_utils import evaluate_model, read_img, to_array


def patch_attack(device, image, applied_patch, mask, target_label, probability_threshold, model, lr=1e-1,
                 max_iteration=200):
    model.eval()
    image_squeezed = image.squeeze(0).float().to(device)
    applied_patch = nn.Parameter(torch.from_numpy(applied_patch).float().to(device), requires_grad=True)
    mask = torch.from_numpy(mask).float().to(device)
    target_probability, count = 0, 0
    optimizer = optim.Adam([applied_patch], lr=lr)
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        perturbated_image = ((1 - mask) * image_squeezed) + (mask * applied_patch)
        optimizer.zero_grad()
        output = model(perturbated_image.float().unsqueeze(0))
        loss = -torch.nn.functional.log_softmax(output, dim=1)[0][target_label]
        loss.backward()
        optimizer.step()
        applied_patch.data = torch.clamp(applied_patch.data, min=-3, max=3).to(device)
        perturbated_image = ((1 - mask) * image_squeezed) + (mask * applied_patch)
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        output = model(perturbated_image.float().unsqueeze(0))
        target_probability = torch.nn.functional.softmax(output, dim=1)[0][target_label].item()

        if count % 10 == 0:
            print(f"Iteration {count}, Target Probability: {target_probability}")

    perturbated_image_t = perturbated_image.detach().clone()
    perturbated_image = perturbated_image.data.cpu().numpy()
    applied_patch = applied_patch.data.cpu().numpy()
    return perturbated_image, applied_patch, perturbated_image_t


#                                       without optimization
# def patch_attack(device, image, applied_patch, mask, target_label, probability_threshold, model, lr=1,
#                  max_iteration=100):
#     model.eval()
#     image_squeezed = image.squeeze(0).float().to(device)
#     applied_patch = torch.from_numpy(applied_patch).float().to(device)
#     mask = torch.from_numpy(mask).float().to(device)
#     target_probability, count = 0, 0
#     perturbated_image = (mask * applied_patch + (1 - mask) * image_squeezed)
#     while target_probability < probability_threshold and count < max_iteration:
#         count += 1
#         perturbated_image.requires_grad = True
#         output = model(perturbated_image.float().unsqueeze(0))
#         target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target_label]
#         target_log_softmax.backward()
#         patch_grad = perturbated_image.grad.clone().detach().to(device)
#         perturbated_image.grad.data.zero_()
#         applied_patch = lr * patch_grad + applied_patch
#         applied_patch = torch.clamp(applied_patch, min=-3, max=3).to(device)
#         perturbated_image = mask * applied_patch + (1 - mask) * image_squeezed
#         perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
#         output = model(perturbated_image.float().unsqueeze(0))
#         target_probability = torch.nn.functional.softmax(output, dim=1)[0][target_label].item()
#     perturbated_image_t = perturbated_image.detach().clone()
#     perturbated_image = to_array(perturbated_image)
#     applied_patch = to_array(applied_patch)
#     return perturbated_image, applied_patch, perturbated_image_t


def patch_train(device, epochs_num, train_loader, model, labels, target_label, test_loader, patch_type, patch,
                probability_threshold, lr, max_iteration):
    model.eval()
    for epoch in range(epochs_num):
        total, success = 0, 0
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for (image, label) in train_loader:
            assert image.shape[0] == 1, 'Обработка только по 1 изображению'
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] != label and predicted[0].data != target_label:
                continue
            total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch,
                                                                          image.shape[1:])
            perturbated_image, applied_patch, perturbated_image_t = patch_attack(device, image, applied_patch, mask,
                                                                                 target_label,
                                                                                 probability_threshold, model, lr,
                                                                                 max_iteration)
            output = model(perturbated_image_t.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data == target_label:
                success += 1
                # plt.imshow(np.clip(np.transpose(perturbated_image, (1, 2, 0)) * std + mean, 0, 1))
                # plt.title(labels[predicted[0].data])
                # plt.show()
            patch = applied_patch[:, x_location:x_location + patch.shape[1],
                    y_location:y_location + patch.shape[2]]
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig("training_pictures/" + str(epoch) + " patch.png")
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch,
                                                                               100 * success / total))
        # train_success_rate = test_patch(device, patch_type, target_label, patch, train_loader, model, labels)
        # print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(device, patch_type, target_label, patch, test_loader, model, labels)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # if test_success_rate > best_patch_success_rate:
        #     best_patch_success_rate = test_success_rate
        #     best_patch_epoch = epoch
        #     plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        #     plt.savefig("training_pictures/best_patch.png")


def main():
    image_path = '../images/img.png'
    img, img_size = read_img(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    # labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    target_label = 55
    train_loader, test_loader = load_train_teat_data(device, r'../dataset/labels.csv', r'../dataset/images')
    # accuracy, pred_labels, false_pred, confidences = evaluate_model(model, test_loader)
    patch = init_patch('rectangle', image_size=(3, 224, 224), noise_percentage=0.035)
    applied_patch, mask, x_location, y_location = mask_generation('rectangle', patch, img.shape[1:])
    perturbated_image, applied_patch, perturbated_image_t = patch_attack(device, img, applied_patch, mask, target_label,
                                                                         0.95, model,
                                                                         max_iteration=200)
    # patch_train(device, 1, train_loader, model, labels, target_label, test_loader, 'rectangle', patch,
    #             0.85, 1, 300)
    print(f'Target label: {labels[target_label]}')
    # print(f'Accuracy model on test dataset: {accuracy}')
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    orig_class = model(img).argmax().item()
    new_class = model(perturbated_image_t.unsqueeze(0)).argmax().item()
    axs[0].imshow(to_array(img))
    axs[0].set_title(f'Original class: {labels[orig_class]}')
    axs[0].axis('off')
    axs[1].imshow(to_array(perturbated_image_t))
    axs[1].set_title(f'Class: {labels[new_class]}')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
