import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from BIM import bim
from FGSM import fgsm
from I_FGSM import i_fgsm
from L_BFGS import l_bfgs
from PGD import pgd
from load_ds import load_data
from utils import evaluate_model, to_array
import torchvision.models as models
from tqdm import tqdm


def apply_fgsm_to_data_loader(model, orig_data, epss, device):
    model.eval()
    results = []
    for eps in epss:
        adversarial_images = []
        true_labels = []
        print(f"\nEps = {eps}")
        for img_tensor, label in tqdm(orig_data):
            img_tensor = img_tensor.to(device)
            adv_images = fgsm(model, img_tensor, eps)
            adversarial_images.append(adv_images)
            true_labels.append(label)
        adversarial_images_tensor = torch.cat(adversarial_images)
        true_labels_tensor = torch.tensor(true_labels).to(device)
        adv_dataset = TensorDataset(adversarial_images_tensor, true_labels_tensor)
        adversarial_data = DataLoader(adv_dataset, batch_size=1)
        results.append(adversarial_data)
    return results


def apply_i_fgsm_to_data_loader(model, orig_data, epss, num_iter, device):
    model.eval()
    results = []
    for eps in epss:
        adversarial_images = []
        true_labels = []
        print(f"\nEps = {eps}")
        for img_tensor, label in tqdm(orig_data):
            img_tensor = img_tensor.to(device)
            adv_images = i_fgsm(model, img_tensor, eps, num_iter)
            adversarial_images.append(adv_images)
            true_labels.append(label)
        adversarial_images_tensor = torch.cat(adversarial_images)
        true_labels_tensor = torch.tensor(true_labels).to(device)
        adv_dataset = TensorDataset(adversarial_images_tensor, true_labels_tensor)
        adversarial_data = DataLoader(adv_dataset, batch_size=1)
        results.append(adversarial_data)
    return results


def apply_bim_to_data_loader(model, orig_data, epss, alph, num_iter, device):
    model.eval()
    results = []
    for eps in epss:
        adversarial_images = []
        true_labels = []
        print(f"\nEps = {eps}")
        print(f"Alpha = {alph}")
        for img_tensor, label in tqdm(orig_data):
            img_tensor = img_tensor.to(device)
            adv_images = bim(model, img_tensor, eps, alph, num_iter, device)
            adversarial_images.append(adv_images)
            true_labels.append(label)
        adversarial_images_tensor = torch.cat(adversarial_images)
        true_labels_tensor = torch.tensor(true_labels).to(device)
        adv_dataset = TensorDataset(adversarial_images_tensor, true_labels_tensor)
        adversarial_data = DataLoader(adv_dataset, batch_size=1)
        results.append(adversarial_data)
    return results


def apply_pgd_to_data_loader(model, orig_data, epss, alph, num_iter, device):
    model.eval()
    results = []
    for eps in epss:
        adversarial_images = []
        true_labels = []
        print(f"\nEps = {eps}")
        print(f"Alpha = {alph}")
        for img_tensor, label in tqdm(orig_data):
            img_tensor = img_tensor.to(device)
            adv_images = pgd(model, img_tensor, eps, alph, num_iter, device)
            adversarial_images.append(adv_images)
            true_labels.append(label)
        adversarial_images_tensor = torch.cat(adversarial_images)
        true_labels_tensor = torch.tensor(true_labels).to(device)
        adv_dataset = TensorDataset(adversarial_images_tensor, true_labels_tensor)
        adversarial_data = DataLoader(adv_dataset, batch_size=1)
        results.append(adversarial_data)
    return results


def apply_l_bfgs_to_data_loader(model, orig_data, target_label, epss, num_iter, device):
    model.eval()
    results = []
    for eps in epss:
        adversarial_images = []
        true_labels = []
        print(f"\nEps = {eps}")
        count = 0
        for img_tensor, label in tqdm(orig_data):
            img_tensor = img_tensor.to(device)
            adv_images = l_bfgs(model, img_tensor, target_label, eps, num_iter, device)
            adversarial_images.append(adv_images)
            true_labels.append(label)
            count += 1
            if count >= 5:
                break
        adversarial_images_tensor = torch.cat(adversarial_images)
        true_labels_tensor = torch.tensor(true_labels).to(device)
        adv_dataset = TensorDataset(adversarial_images_tensor, true_labels_tensor)
        adversarial_data = DataLoader(adv_dataset, batch_size=1)
        results.append(adversarial_data)
    return results


def plot_five_img_orig(data, title, labels):
    counter = 0
    plt.figure(figsize=(25, 7))
    for img_tensor, label in data:
        if counter >= 5:
            break
        adv_img = to_array(img_tensor.cpu())
        plt.subplot(1, 5, counter + 1)
        plt.imshow(adv_img)
        plt.title(f'Label: {labels[label.item()]}')
        plt.axis('off')
        counter += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_five_img_adv(data, pred_labels, title, labels):
    counter = 0
    plt.figure(figsize=(25, 7))
    for (img_tensor, _), label in zip(data, pred_labels):
        if counter >= 5:
            break
        adv_img = to_array(img_tensor.cpu())
        plt.subplot(1, 5, counter + 1)
        plt.imshow(adv_img)
        plt.title(f'Label: {labels[label.item()]}')
        plt.axis('off')
        counter += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_acc_diff(accuracy, epss):
    eps_plot = [0] + epss
    plt.plot(eps_plot, accuracy, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.show()


def fgsm_analitics(device):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    epss = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    adv_data = apply_fgsm_to_data_loader(model, orig_data, epss, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred, change_label_num = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps})", labels)
        print(f'Точность модели на тестовом наборе при eps={eps} : {accuracy:.2f}')
        print(f'Количество изображений, поменявших свой класс: {change_label_num}')
    plot_acc_diff(acc_list, epss)


def i_fgsm_analitics(device):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    # plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    epss = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
    adv_data = apply_i_fgsm_to_data_loader(model, orig_data, epss, 10, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred, change_label_num = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps}, iter={10})", labels)
        print(f'Точность модели на тестовом наборе при eps={eps} и 10 итерациях: {accuracy:.2f}')
        print(f'Количество изображений, поменявших свой класс: {change_label_num}')
    plot_acc_diff(acc_list, epss)


def bim_analitics(device):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    # plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    epss = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
    alpha = 0.005  # 0.001, 0.002, 0.005
    adv_data = apply_bim_to_data_loader(model, orig_data, epss, alpha, 10, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred, change_label_num = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps}, iter={10}), alpha={alpha}", labels)
        print(f'Точность модели на тестовом наборе при eps={eps}, и 10 итерациях: {accuracy:.2f}')
        print(f'Количество изображений, поменявших свой класс: {change_label_num}')
    plot_acc_diff(acc_list, epss)


def pgd_analitics(device):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    # plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    # epss = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    epss = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    alpha = 0.15  # [0.01, 0.05, 0.1, 0.15]
    adv_data = apply_pgd_to_data_loader(model, orig_data, epss, alpha, 10, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred, change_label_num = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps}, iter={10}), alpha={alpha}", labels)
        print(f'Точность модели на тестовом наборе при eps={eps}, alpha={alpha} и 10 итерациях: {accuracy:.2f}')
        print(f'Количество изображений, поменявших свой класс: {change_label_num}')
    plot_acc_diff(acc_list, epss)


def l_bfgs_analitics(device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    labels = models.ResNet50_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    # plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    # epss = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
    epss = [50]
    target_label = 81
    num_iter = 10
    adv_data = apply_l_bfgs_to_data_loader(model, orig_data, target_label, epss, num_iter, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred, change_label_num = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps}, iter={num_iter})", labels)
        print(f'Точность модели на тестовом наборе при eps={eps}, и 10 итерациях: {accuracy:.2f}')
        print(f'Количество изображений, поменявших свой класс: {change_label_num}')
    plot_acc_diff(acc_list, epss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l_bfgs_analitics(device)


if __name__ == "__main__":
    main()
