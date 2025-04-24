import os
import shutil
from pycocotools.coco import COCO
import json

annotation_file = r"C:\Users\vodnyy\Downloads\COCO\annotations\instances_val2017.json"  # или val2017
images_dir = r"C:\Users\vodnyy\Downloads\COCO\val2017"  # или val2017
output_dir  = "C:/Users/vodnyy/Downloads/coco_person/val2017/"  # Папка для сохранения
output_annotations_path = "C:/Users/vodnyy/Downloads/coco_person/person_val2017.json"

os.makedirs(output_dir , exist_ok=True)
os.makedirs(os.path.dirname(output_annotations_path), exist_ok=True)

# coco = COCO(annotations_path)
# cat_ids = coco.getCatIds(catNms=['person'])
# assert cat_ids, "Класс 'person' не найден в аннотациях!"
#
# img_ids = coco.getImgIds(catIds=cat_ids)     # ID изображений с людьми
#
# filtered_data = {
#     "info": coco.dataset.get("info", {}),
#     "licenses": coco.dataset.get("licenses", []),
#     "categories": [coco.loadCats(cat_ids)[0]],  # Только категория "person"
#     "images": [],
#     "annotations": []
# }
#
# for img_id in img_ids:
#     # Загружаем информацию об изображении
#     img_info = coco.loadImgs(img_id)[0]
#
#     # Копируем изображение
#     src_path = os.path.join(images_dir, img_info['file_name'])
#     dst_path = os.path.join(output_images_dir, img_info['file_name'])
#     shutil.copy(src_path, dst_path)
#
#     # Добавляем информацию об изображении в новый JSON
#     filtered_data["images"].append(img_info)
#
#     # Добавляем все аннотации для людей в этом изображении
#     ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
#     annotations = coco.loadAnns(ann_ids)
#     filtered_data["annotations"].extend(annotations)
#
# # Сохраняем отфильтрованные аннотации
# with open(output_annotations_path, 'w') as f:
#     json.dump(filtered_data, f)
#
# print(f"Готово! Отфильтровано {len(img_ids)} изображений.")
# print(f"Изображения сохранены в: {output_images_dir}")
# print(f"Аннотации сохранены в: {output_annotations_path}")

coco = COCO(annotation_file)
person_id = coco.getCatIds(catNms=['person'])[0]

# Собираем ID изображений ТОЛЬКО с людьми
pure_person_img_ids = []
for img_id in coco.getImgIds():
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Проверяем, что все аннотации в изображении - люди
    if all(ann['category_id'] == person_id for ann in annotations) and annotations:
        pure_person_img_ids.append(img_id)

# Создаем новые аннотации
filtered_data = {
    "info": coco.dataset.get("info", {}),
    "licenses": coco.dataset.get("licenses", []),
    "categories": [coco.loadCats([person_id])[0]],
    "images": [],
    "annotations": []
}

# Копируем изображения и собираем аннотации
for img_id in pure_person_img_ids:
    img_info = coco.loadImgs(img_id)[0]
    shutil.copy(
        os.path.join(images_dir, img_info['file_name']),
        os.path.join(output_dir, img_info['file_name'])
    )

    filtered_data["images"].append(img_info)
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_id])
    filtered_data["annotations"].extend(coco.loadAnns(ann_ids))

# Сохраняем новые аннотации
with open(os.path.join(output_dir, "annotations.json"), "w") as f:
    json.dump(filtered_data, f)

print(f"Найдено {len(pure_person_img_ids)} изображений только с людьми")
print(f"Данные сохранены в: {output_dir}")