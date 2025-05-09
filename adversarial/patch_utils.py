import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import median_pool


def generate_patch(patch_size, device, mode):
    if mode == "gray":
        return torch.full((3, patch_size, patch_size), 0.5).to(device)
    if mode == "rand":
        return torch.rand((3, patch_size, patch_size)).to(device)


def create_print_ability_tensor(patch_side):
    # Набор триплетов для оптимальной печати патча
    printable_rgb_triplets = [
        [0.10, 0.10, 0.10],  # Чёрный (90% насыщенности)
        [0.20, 0.20, 0.20],  # Тёмно-серый
        [0.05, 0.05, 0.15],  # Тёмно-синий
        [0.15, 0.05, 0.05],  # Тёмно-красный
        [0.07, 0.10, 0.07],  # Тёмно-зелёный
        [0.70, 0.10, 0.10],  # Насыщенный красный
        [0.90, 0.30, 0.30],  # Светло-красный
        [0.80, 0.20, 0.50],  # Пурпурный
        [0.95, 0.50, 0.50],  # Пастельно-розовый
        [0.20, 0.50, 0.20],  # Травяной зелёный
        [0.40, 0.70, 0.30],  # Салатовый
        [0.10, 0.30, 0.10],  # Тёмно-зелёный
        [0.60, 0.80, 0.50],  # Мятный
        [0.10, 0.10, 0.60],  # Тёмно-синий
        [0.30, 0.30, 0.90],  # Ярко-синий
        [0.50, 0.70, 0.90],  # Голубой
        [0.20, 0.50, 0.70],  # Морская волна
        [0.90, 0.80, 0.10],  # Золотистый
        [0.95, 0.60, 0.10],  # Оранжевый
        [0.80, 0.70, 0.20],  # Горчичный
        [0.50, 0.20, 0.70],  # Фиолетовый
        [0.70, 0.40, 0.80],  # Лавандовый
        [0.40, 0.30, 0.10],  # Коричневый
        [0.60, 0.40, 0.20],  # Терракота
        [0.95, 0.95, 0.95],  # Белый (5% серого)
        [0.90, 0.90, 0.80],  # Кремовый
        [0.70, 0.80, 0.90],  # Светло-голубой
        [0.90, 0.70, 0.70],  # Розовый песок
        [0.85, 0.10, 0.10],  # Epson Red
        [0.10, 0.60, 0.20],  # Canon Green
        [0.20, 0.20, 0.80]  # HP Blue
    ]
    color_array = np.array(printable_rgb_triplets, dtype=np.float32)
    print_ability_tensors = []
    for color in color_array:
        color_tensor = np.tile(
            color.reshape(3, 1, 1),  # Исходный цвет [3] -> [3, 1, 1]
            (1, patch_side, patch_side)  # Размножаем по spatial-размерностям
        )
        print_ability_tensors.append(color_tensor)

    print_ability_array = np.stack(print_ability_tensors, axis=0)

    return torch.from_numpy(print_ability_array)


def calc_nps(patch, print_ability_tensor):
    c_dist = (patch - print_ability_tensor + 0.000001) ** 2
    c_dist = torch.sum(c_dist, 1) + 0.000001
    c_dist = torch.sqrt(c_dist)
    c_dist_min = torch.min(c_dist, 0)[0]
    nps_score = torch.sum(c_dist_min)
    return nps_score / torch.numel(patch)



def calc_total_variation(patch):
    diff_h = torch.sum(torch.abs(patch[:, :, 1:] - patch[:, :, :-1] + 0.000001), 0)
    diff_h = torch.sum(torch.sum(diff_h, 0), 0)
    diff_v = torch.sum(torch.abs(patch[:, 1:, :] - patch[:, :-1, :] + 0.000001), 0)
    diff_v = torch.sum(torch.sum(diff_v, 0), 0)
    total_variation = diff_h + diff_v
    return total_variation / torch.numel(patch)


def max_prob_extraction(model_outputs, cls_id, num_cls):
    batch_probs = []
    for output in model_outputs:
        batch, anchors, grid_h, grid_w, _ = output.shape
        output = output.view(
            batch,
            anchors * grid_h * grid_w,
            5 + num_cls
        )
        obj_scores = torch.sigmoid(output[..., 4])
        cls_output = output[..., 5:5 + num_cls]
        cls_probs = torch.softmax(cls_output, dim=2)
        target_cls_probs = cls_probs[..., cls_id]
        combined_probs = obj_scores #* target_cls_probs

        max_probs, _ = combined_probs.max(dim=1)
        batch_probs.append(max_probs)

    final_probs = torch.stack(batch_probs).max(dim=0).values
    return final_probs


def transform_patch(device, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
    min_contrast = 0.8
    max_contrast = 1.2
    min_brightness = -0.1
    max_brightness = 0.1
    noise_factor = 0.10
    min_angle = -20 / 180 * math.pi
    max_angle = 20 / 180 * math.pi
    median_pooler = median_pool.MedianPool2d(7, same=True)

    adv_patch = median_pooler(adv_patch.unsqueeze(0))
    pad = (img_size - adv_patch.size(-1)) / 2
    adv_patch = adv_patch.unsqueeze(0)
    adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
    batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

    contrast = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(min_contrast, max_contrast)
    contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    brightness = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(min_brightness, max_brightness)
    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    noise = torch.empty(adv_batch.size(), device=device, dtype=torch.float32).uniform_(-1, 1) * noise_factor

    adv_batch = adv_batch * contrast + brightness + noise
    adv_batch = torch.clamp(adv_batch, 1e-6, 0.99999)

    # Создание маски: патч применяется только к class_id == 0
    cls_ids = lab_batch.narrow(2, 0, 1)
    cls_mask = (cls_ids == 0).float()
    cls_mask = cls_mask.expand(-1, -1, 3).unsqueeze(-1).unsqueeze(-1)
    cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3), adv_batch.size(4))
    msk_batch = cls_mask

    # Паддинг
    mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
    adv_batch = mypad(adv_batch)
    msk_batch = mypad(msk_batch)

    # Подготовка к аффинному преобразованию
    anglesize = lab_batch.size(0) * lab_batch.size(1)
    if do_rotate:
        angle = torch.empty(anglesize, device=device, dtype=torch.float32).uniform_(min_angle, max_angle)
    else:
        angle = torch.zeros(anglesize, device=device, dtype=torch.float32)

    current_patch_size = adv_patch.size(-1)
    lab_batch_scaled = torch.zeros_like(lab_batch, device=device, dtype=torch.float32)
    lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
    lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
    lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
    lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

    target_size = torch.sqrt((lab_batch_scaled[:, :, 3] * 0.2) ** 2 + (lab_batch_scaled[:, :, 4] * 0.2) ** 2)

    target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
    target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
    target_off_x = lab_batch[:, :, 3].view(np.prod(batch_size))
    target_off_y = lab_batch[:, :, 4].view(np.prod(batch_size))

    if rand_loc:
        off_x = target_off_x * torch.empty_like(target_off_x).uniform_(-0.4, 0.4)
        off_y = target_off_y * torch.empty_like(target_off_y).uniform_(-0.4, 0.4)
        target_x = target_x + off_x
        target_y = target_y + off_y

    target_y = target_y - 0.05
    scale = (target_size / current_patch_size).reshape(anglesize)

    s = adv_batch.size()
    adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
    msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

    tx = (-target_x + 0.5) * 2
    ty = (-target_y + 0.5) * 2
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    # Матрица аффинного преобразования с поворотом и масштабом
    theta = torch.zeros(anglesize, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = sin / scale
    theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
    theta[:, 1, 0] = -sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

    grid = F.affine_grid(theta, adv_batch.shape, align_corners=False)
    adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
    msk_batch_t = F.grid_sample(msk_batch, grid, align_corners=False)

    adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
    msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

    adv_batch_t = torch.clamp(adv_batch_t, 1e-6, 0.999999)

    return adv_batch_t * msk_batch_t


def apply_patch_to_img_batch(img_batch, adv_batch):
    advs = torch.unbind(adv_batch, 1)
    for adv in advs:
        img_batch = torch.where((adv == 0), img_batch, adv)
    return img_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_img = generate_patch(300, device, "gray")
    loaded_tensor = torch.load("./print_ability_tensor.pt").requires_grad_(False)
    img_size = 640
    nps = calc_nps(adv_img, loaded_tensor)


if __name__ == '__main__':
    main()
