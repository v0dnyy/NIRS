## Постановка задачи:
Разработка методов генерации adversarial примеров и оценка их воздействия на предобученные нейросетевые модели для классификации и обнаружения людей, с акцентом на анализ изменений в точности(accuracy) моделей при использовании как стандартного тестового набора данных, так и набора данных, состоящего из adversarial примеров, сгененрированных с помощью различных методов атак при различных гиперпараметрах.


### Иследуемые предобученные модели.
Для анализа использовались модели, предобученные на наборе данных ImageNet, из модуля [torchvision](https://pytorch.org/vision/stable/) библиотеки [PyTorch](https://pytorch.org/).

Для задачи обнаружения объектов в модуле доступны следующие модели:
- [**Faster R-CNN**](https://pytorch.org/vision/stable/models/faster_rcnn.html);
- [**RetinaNet**](https://pytorch.org/vision/stable/models/retinanet.html);
- [**SSD**](https://pytorch.org/vision/stable/models/ssd.html);
- [**SSDlite**](https://pytorch.org/vision/stable/models/ssdlite.html).

В backbone моделей, приведенных выше, используютя следующие модели классификации:
- [**ResNet50**](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) с механизмом Feature Pyramid Network(FPN) для более эффективного извлечения признаков;
- [**MobileNet V3**](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.mobilenet_v3_large);
- [**VGG-16**](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16).

Было принято решение использовать предобученные на наборе данных [ImageNet](https://image-net.org/index.php) модели классификации, представляющие backbone моделей обнаружения объектов, для проведения исследоваения.


### Тестовый набор данных.
Для оценки точности предобученнах моделей, был собран и аннотирован тестовый набор данных, состоящий из 200 изображений различных классов набора данных ImageNet.

Набор данных представлен в репозитории -- [тестовый набор данных](dataset/)

Папка [images](dataset/images/) содержит изображения, входящие в тестовый набор данных.
Файл [labels.csv](dataset/labels.csv) содержит в себе аннотации к изображениям.  


### Оценка точности предобученных моделей на тестовом наборе данных.
- ResNet50
Точность модели на тестовом наборе: 96.00 %
- MobileNet V3
Точность модели на тестовом наборе: 89.50 %
- VGG-16
Точность модели на тестовом наборе: 83.00
