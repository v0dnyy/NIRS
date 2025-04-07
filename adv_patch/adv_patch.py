import torch
from torchvision import models

from load_ds import load_train_teat_data
from utils import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()
    labels = models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    target_label = 52
    train_loader, test_loader = load_train_teat_data(device, r'../dataset/labels.csv', r'../dataset/images')
    accuracy, pred_labels, false_pred, confidences = evaluate_model(model, test_loader)
    print(f'Target label: {labels[target_label]}')
    print(f'Accuracy model on test dataset: {accuracy}')


if __name__ == '__main__':
    main()