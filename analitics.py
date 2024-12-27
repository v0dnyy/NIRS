import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from FGSM import fgsm
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    orig_data = load_data(device)
    plot_five_img_orig(orig_data, "Original Images", labels)
    acc_list = []
    accuracy = evaluate_model(model, orig_data)[0]
    acc_list.append(accuracy)
    print(f'Точность модели на тестовом наборе: {accuracy:.2f}')
    # epss = [0.1, 0.2, 0.3]
    epss = [0.1]
    adv_data = apply_fgsm_to_data_loader(model, orig_data, epss, device)
    for eps, adv in zip(epss, adv_data):
        accuracy, pred = evaluate_model(model, adv)
        acc_list.append(accuracy)
        plot_five_img_adv(adv, pred, f"Adv Images(eps = {eps})", labels)
        print(f'Точность модели на тестовом наборе при eps={eps} : {accuracy:.2f}')
    plot_acc_diff(acc_list,epss)

if __name__ == "__main__":
    main()
