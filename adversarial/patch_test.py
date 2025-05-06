import torch
from dataset import *
from ultralytics import YOLO

def test_model(device, model, img_dir, labels_dir, batch_size):
    # test_loader = torch.utils.data.DataLoader(PersonDataset(img_dir, labels_dir, 640,
    #                                                          shuffle=True),
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4)
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
    test_img_dir = '../adversarial/dataset/test/images'
    test_labels_dir = '../adversarial/dataset/test/labels'
    model = YOLO("yolov5s.pt")
    model.info()
    metrics = model.val(data = "../adversarial/dataset/data.yaml")
    # print(metrics.box.map)  # map50-95
    # print(metrics.box.map50)  # map50
    # print(metrics.box.map75)  # map75
    # print(metrics.box.maps)  # a list contains map50-95 of each category


if __name__ == "__main__":
    main()
