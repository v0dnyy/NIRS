import torch
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.general import non_max_suppression, scale_boxes, check_dataset, xywh2xyxy, box_iou, coco80_to_coco91_class
from yolov5.utils.dataloaders import create_dataloader
import numpy as np
import os


def evaluate_yolov5(model, data_path, conf_thres, iou_thres, device, selected_classes):
    model.eval()
    data = check_dataset(data_path)
    dataloader, dataset = create_dataloader(data['test'], imgsz=640, batch_size=8, stride=model.stride, rect=True, prefix='test: ')

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # IoU от 0.5 до 0.95
    niou = iouv.numel()
    stats = []


    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        targets = targets.to(device)
        with torch.no_grad():
            pred = model(img)  # Получаем все выходы
            if isinstance(pred, (list, tuple)):
                pred = pred[0]  # Берем основной тензор предсказаний
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        for si, pred in enumerate(pred):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Фильтрация по выбранным классам
            if selected_classes is not None:
                pred = pred[np.isin(pred[:, 5].cpu().numpy(), selected_classes)]
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), tcls))
                    continue

            predn = pred.clone()  # копируем предсказания
            scale_boxes(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # индексы уже сопоставленных GT
                tcls_tensor = labels[:, 0]
                # Конвертация GT в xyxy формат
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # масштабирование GT
                for j in range(nl):
                    iou = box_iou(predn[:, :4], tbox[j].unsqueeze(0))
                    iou, i = iou.max(0)

                    print(f"\nОбработка GT {j}:")
                    print(f"Класс GT: {tcls_tensor[j].item()}")
                    print(f"Координаты GT: {tbox[j]}")
                    print(f"Лучшее совпадение - pred {i}: класс {pred[i, 5].item()}, IoU {iou.item()}")
                    print(f"Уже обнаружено: {i in detected}")

                    if iou >= iouv[0]:
                        print("IoU >= порога")
                        if pred[i, 5] == tcls_tensor[j]:
                            print("Классы совпадают")
                            if i not in detected:
                                print("Новое обнаружение!")
                                correct[i] = iou >= iouv
                                detected.append(i)
                        else:
                            print("Классы не совпадают")
                    else:
                        print("IoU ниже порога")

                    if iou >= iouv[0] and pred[i, 5] == tcls_tensor[j] and i not in detected:
                        correct[i] = iou >= iouv  # правильное для всех порогов IoU
                        detected.append(i)
            stats.append((correct.cpu(),
                         pred[:, 4].cpu(),  # confidence
                         pred[:, 5].cpu(),  # predicted class
                         tcls))  # target class

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        # Фильтрация результатов по выбранным классам
        if selected_classes is not None:
            mask = np.isin(ap_class, selected_classes)
            ap_class = ap_class[mask]
            ap50, ap = ap50[mask], ap[mask]
            p, r = p[mask], r[mask]  # Фильтруем precision и recall
            print(f"Оценка только для классов: {ap_class.tolist()}")

        print(f"mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")

        # Создаем словарь с метриками для каждого класса
        class_metrics = {}
        for i, class_id in enumerate(ap_class):
            class_metrics[class_id] = {
                'class_name': model.names[class_id],
                'precision': p[i],
                'recall': r[i],
                'ap50': ap50[i],
                'ap': ap[i]
            }

        # Возвращаем все метрики
        return {
            'map50': map50,
            'map': map,
            'class_metrics': class_metrics,
            'global_precision': mp,
            'global_recall': mr
        }
    else:
        print("Нет детекций для оценки")
        return {
            'map50': 0.0,
            'map': 0.0,
            'class_metrics': {},
            'global_precision': 0.0,
            'global_recall': 0.0
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images_dir = "../adversarial/dataset/test/images"
    conf_thres = 0.3
    iou_thres = 0.5
    model_p = torch.hub.load('../yolov5', 'yolov5s', source='local', pretrained=True).eval()
    evaluate_yolov5(model_p,'../adversarial/dataset/data.yaml', conf_thres, iou_thres, device, [0])


if __name__ == "__main__":
    main()
