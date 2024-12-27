import torch
import pandas as pd
from utils import read_img, evaluate_model
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset



def load_data(device, csv_file = 'dataset\labels.csv', img_dir= 'dataset\images', batch_size=1):
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


def main():
    csv_file = "dataset\labels.csv"
    img_dir = "dataset\images"
    data_loader = load_data(csv_file=csv_file, img_dir=img_dir, batch_size=32)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    accuracy = evaluate_model(model, data_loader)
    print('Точность модели на тестовом наборе: {:.2f} %'.format(accuracy))



if __name__ == "__main__":
    main()
