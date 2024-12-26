import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, ssdlite320_mobilenet_v3_large, ssd300_vgg16, \
    retinanet_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights, \
    SSD300_VGG16_Weights, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as f
from PIL import Image
import matplotlib.pyplot as plt


def load_model_weights(model_name: str):
    labels = []
    if model_name == 'F-R-CNN':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights = weights)
    elif model_name == 'SSDLite':
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights = weights)
    elif model_name == 'SSD':
        weights = SSD300_VGG16_Weights.DEFAULT
        model = ssd300_vgg16(weights = weights)
    elif model_name == 'RetinaNet':
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights = weights)
    labels = weights.meta["categories"]
    model.eval()
    return model, labels


def detect_objects(model, image_path):
    image = Image.open(image_path)
    image_tensor = f.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    return image, predictions


def display_results(image, predictions, labels, model_name, threshold=0.5):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    # Отображение предсказанных объектов
    for i in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][i].item()
        label_id = predictions[0]["labels"][i].item()
        if score > threshold:  # Фильтрация по классу человек: and label_id == 1
            box = predictions[0]['boxes'][i].cpu().numpy()
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                              fill=False, color='red', linewidth=2))
            label_id = predictions[0]["labels"][i].item()
            label_name = labels[label_id]
            plt.text(box[0], box[1], f'{label_name}: {score:.2f}',
                     fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.title(model_name)
    plt.axis('off')
    plt.show()


def main():
    image_path = 'res.png'
    # image_path = 'imgs/Panda_closeup.jpg'
    model_lst = ['F-R-CNN','SSDLite', 'SSD', 'RetinaNet']
    for elem in model_lst:
        model, labels = load_model_weights(elem)
        model_name = type(model).__name__
        image, predictions = detect_objects(model, image_path)
        display_results(image, predictions, labels, model_name, 0.7)


if __name__ == '__main__':
    main()
