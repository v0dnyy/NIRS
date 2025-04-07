import torch
import pandas as pd
import numpy as np
from utils import read_img, evaluate_model
import torchvision.models as models
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def load_data(device, csv_file='dataset\labels.csv', img_dir='dataset\images', batch_size=1):
    labels_df = pd.read_csv(csv_file)
    labels_df['ImgId'] = labels_df['ImgId'].astype(str).str.zfill(3)
    images = []
    labels = []
    for idx in range(len(labels_df)):
        img_id = labels_df.iloc[idx]['ImgId']
        label = labels_df.iloc[idx]['TrueLabel']
        img_path = f"{img_dir}/{img_id}.png"
        try:
            img_tensor, _ = read_img(img_path)
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_path}: {e}")
    images_tensor = torch.cat(images).to(device)
    labels_tensor = torch.tensor(labels).to(device)
    dataset = TensorDataset(images_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def load_train_teat_data(device, csv_file=r'dataset/labels.csv', img_dir='dataset/images', batch_size=16, test_size=0.2):
    labels_df = pd.read_csv(csv_file)
    labels_df['ImgId'] = labels_df['ImgId'].astype(str).str.zfill(3)
    images = []
    labels = []
    for idx in range(len(labels_df)):
        img_id = labels_df.iloc[idx]['ImgId']
        label = labels_df.iloc[idx]['TrueLabel']
        img_path = f"{img_dir}/{img_id}.png"
        try:
            img_tensor, _ = read_img(img_path)
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_path}: {e}")
    images_tensor = torch.cat(images).to(device)
    labels_tensor = torch.tensor(labels).to(device)
    dataset = TensorDataset(images_tensor, labels_tensor)
    indices = np.arange(len(labels_df))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = "dataset\labels.csv"
    img_dir = "dataset\images"
    data_loader = load_data(device)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    labels = models.VGG16_Weights.DEFAULT.meta['categories']
    model.eval()
    accuracy, _ = evaluate_model(model, data_loader)
    print('Точность модели на тестовом наборе: {:.2f} %'.format(accuracy))


if __name__ == "__main__":
    main()
