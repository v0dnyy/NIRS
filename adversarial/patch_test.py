import torch
from dataset import *
from yolov5.val import run
import yaml


def test_model(device, model, img_dir, labels_dir, batch_size):
    # test_loader = torch.utils.data.DataLoader(PersonDataset(img_dir, labels_dir, 640,
    #                                                          shuffle=True),
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4)
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_p = torch.hub.load('../yolov5', 'yolov5s', source='local', pretrained=True).eval()
    data_path = "../adversarial/dataset/data.yaml"
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    # metrics = model_p.run(data = "../adversarial/dataset/data.yaml")
    if os.path.exists(data['val']):
        print("Путь к валидационному датасету корректен")
    else:
        print("Путь к валидационному датасету неверный:", data['val'])
    metrics = run(data=data, weights="../adversarial/yolov5s.pt", single_cls=True)
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category


if __name__ == "__main__":
    main()
