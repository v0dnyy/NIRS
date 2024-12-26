from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights, VGG16_Weights, vgg16
import torch


def predict_image_class(image_path):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    img = decode_image(image_path)
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model to make a prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(batch).squeeze(0).softmax(0)

    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]

    return f"{category_name, class_id}: {100 * score:.1f}%"


# Пример использования функции
# image_path = "imgs/Panda_closeup.jpg"
image_path = "orig.png"
adv = 'l_bfgs.png'
result = predict_image_class(image_path)
print(result)
result = predict_image_class(adv)
print(result)
