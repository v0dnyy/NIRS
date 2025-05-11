import torch
import numpy as np
import patch_utils
from yolov5.utils.metrics import ap_per_class, box_iou
from yolov5.utils.general import non_max_suppression, scale_boxes, check_dataset, xywh2xyxy
from yolov5.utils.dataloaders import create_dataloader
from PIL import Image
from torchvision import transforms





def filter_pred_by_classes(pred, selected_classes, device):
    class_probs = pred[..., 5:]
    class_ids = class_probs.argmax(dim=2)
    selected_classes_tensor = torch.tensor(selected_classes, device=device)
    mask = torch.zeros_like(class_ids, dtype=torch.bool)
    for cls in selected_classes_tensor:
        mask |= (class_ids == cls)
    pred_filtered = []
    batch_size = pred.shape[0]
    for i in range(batch_size):
        pred_i = pred[i]
        mask_i = mask[i]
        pred_filtered.append(pred_i[mask_i])
    return pred_filtered


def evaluate_yolov5(model, data_path, conf_thres, iou_thres, device, selected_classes = None, apply_patch = False, patch_path = None):
    model.eval()
    data = check_dataset(data_path)
    dataloader, dataset = create_dataloader(data['test'], imgsz=640, batch_size=8, stride=model.stride, rect=True, prefix='test: ', shuffle=True)
    if apply_patch:
        patch_img = Image.open(patch_path).convert('RGB')
        adv_patch = transforms.ToTensor()(patch_img).to(device)
        img_size = 640
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    stats = []

    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)

        if apply_patch:
            batch_size = img.size(0)
            lab_batch = patch_utils.targets_to_lab_batch(targets, batch_size, device)
            adv_batch = patch_utils.patch_to_batch(device, adv_patch, lab_batch, img_size)
            img = patch_utils.apply_patch_to_img_batch(img, adv_batch)
            # p_img = img[1, :, :]
            # p_img = transforms.ToPILImage('RGB')(p_img.detach().cpu())
            # p_img.show()

        with torch.no_grad():
            pred = model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        if selected_classes is not None:
            pred = filter_pred_by_classes(pred, selected_classes, device)
        else:
            pred = [pred[i] for i in range(pred.shape[0])]

        pred = [non_max_suppression(p.unsqueeze(0), conf_thres, iou_thres)[0] for p in pred]

        for si, pred_i  in enumerate(pred):
            batch_idx = targets[:, 0] == si
            labels = targets[batch_idx, 1:]
            gt_num = len(labels)
            gt_cls = labels[:, 0].tolist() if gt_num else []

            if len(pred_i) == 0:
                if gt_num:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), gt_cls))
                continue

            predn = pred_i.clone()
            img0_shape = tuple(shapes[si][0])
            ratio_pad = (tuple(shapes[si][1][0]), tuple(shapes[si][1][1]))
            scale_boxes(img[si].shape[1:], predn[:, :4], img0_shape, ratio_pad)

            correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool, device=device)
            if gt_num:
                detected = []
                gt_cls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, 0] *= img0_shape[1]
                tbox[:, 1] *= img0_shape[0]
                tbox[:, 2] *= img0_shape[1]
                tbox[:, 3] *= img0_shape[0]
                for j in range(gt_num):
                    iou = box_iou(predn[:, :4], tbox[j].unsqueeze(0))
                    iou, i = iou.max(0)
                    if iou >= iouv[0] and predn[i, 5] == gt_cls_tensor[j] and i not in detected:
                        correct[i] = iou >= iouv
                        detected.append(i)

            stats.append((correct.cpu(),
                          predn[:, 4].cpu(),  # confidence
                          predn[:, 5].cpu(),  # predicted class
                          gt_cls))  # target class

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else [np.array([])]*4
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=model.names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()

        # Фильтрация результатов по выбранным классам
        if selected_classes is not None:
            mask = np.isin(ap_class, selected_classes)
            ap_class = ap_class[mask]
            ap50, ap = ap50[mask], ap[mask]
            p, r = p[mask], r[mask]  # Фильтруем precision и recall
            print(f"Оценка только для классов: {ap_class.tolist()}")

        print(f"P: {p}, R: {r}, mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {mAP:.4f}")

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
            'map': mAP,
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
    conf_thres = 0.5
    iou_thres = 0.5
    # model_p = torch.hub.load('../yolov5', 'yolov5s', source='local', pretrained=True).eval()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
    metrics = evaluate_yolov5(
        model = model,
        data_path = '../adversarial/dataset/data.yaml',
        conf_thres = conf_thres,
        iou_thres = iou_thres,
        device = device,
        selected_classes = [0],
        apply_patch = False,
        patch_path = "../adversarial/rand_init_only_obj/patch_obj_e_1500_b_8_tv_2.5_nps_0.01.png"
    )
    print("END")


if __name__ == "__main__":
    main()
