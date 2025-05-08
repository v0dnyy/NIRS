import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler
import os
from datetime import datetime
import torchvision
import patch_utils
from dataset import PersonDataset


def train(device, img_dir, labels_dir, patch_size, patch_mode, batch_size, epochs_num, max_labels, model, nps_coef,
          tv_coef):
    scaler = GradScaler(device=device.type)
    log_dir = os.path.join("../adversarial/logs", (f"e_{epochs_num}_b_{batch_size}_tv_{tv_coef}_nps_{nps_coef}" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")))
    writer = SummaryWriter(log_dir=log_dir)
    start_learning_rate = 0.05
    adv_patch = patch_utils.generate_patch(patch_size, device, patch_mode).requires_grad_(True)
    print_ability_tensor = patch_utils.create_print_ability_tensor(patch_size).to(device)
    train_loader = torch.utils.data.DataLoader(PersonDataset(img_dir, labels_dir, max_labels, 640,
                                                             shuffle=True),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True if device.type == 'cuda' else False)
    epoch_length = len(train_loader)
    optimizer = optim.Adam([adv_patch], lr=start_learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    for epoch in range(epochs_num):
        epoch_start_time = time.time()
        epoch_detection_loss = 0
        epoch_nps_loss = 0
        epoch_total_variation_loss = 0
        epoch_loss = 0
        for i_batch, (img_batch, label_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                      total=epoch_length):
            optimizer.zero_grad()
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            with autocast(device_type=device.type, dtype=torch.float16):
                adv_batch_transformed = patch_utils.transform_patch(device, adv_patch, label_batch, 640, do_rotate=True,
                                                                    rand_loc=False)
                p_img_batch = patch_utils.apply_patch_to_img_batch(img_batch, adv_batch_transformed)
                # img = p_img_batch[1, :, :]
                # img = torchvision.transforms.ToPILImage('RGB')(img.detach().cpu())
                # img.show()
                raw_outputs = model(p_img_batch)

                max_prob = patch_utils.max_prob_extraction(raw_outputs[1], 0, 80)
                total_variation = patch_utils.calc_total_variation(adv_patch)
                nps = patch_utils.calc_nps(adv_patch, print_ability_tensor)

                nps_loss = nps * nps_coef
                total_variation_loss = total_variation * tv_coef
                detection_loss = max_prob.mean()
                loss = detection_loss + nps_loss + torch.max(total_variation_loss, torch.tensor(0.1).to(device))

            epoch_detection_loss += detection_loss.item()
            epoch_total_variation_loss += total_variation_loss.item()
            epoch_nps_loss += nps_loss.item()
            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            adv_patch.data.clamp_(0, 1)

            del adv_batch_transformed, p_img_batch, raw_outputs, max_prob, detection_loss, nps_loss, total_variation_loss, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        epoch_detection_loss = epoch_detection_loss / len(train_loader)
        epoch_nps_loss = epoch_nps_loss / len(train_loader)
        epoch_total_variation_loss = epoch_total_variation_loss / len(train_loader)
        epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        writer.add_scalar('epoch/total_loss', epoch_loss, epoch)
        writer.add_scalar('epoch/detection_loss', epoch_detection_loss, epoch)
        writer.add_scalar('epoch/nps_loss', epoch_nps_loss, epoch)
        writer.add_scalar('epoch/tv_loss', epoch_total_variation_loss, epoch)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('epoch/time', epoch_time, epoch)
        writer.add_image('epoch/patch', adv_patch, epoch)

        scheduler.step(epoch_loss)

    writer.close()
    img = torchvision.transforms.ToPILImage('RGB')(adv_patch)
    # img.show()
    img.save(f"../adversarial/patch_e_{epochs_num}_b_{batch_size}_tv_{tv_coef}_nps_{nps_coef}.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False).eval()
    train_img_dir = '../adversarial/dataset/train/images'
    train_labels_dir = '../adversarial/dataset/train/labels'
    batch_size = 8
    epochs_num = 1000
    max_labels = 24
    patch_size = 300
    patch_mode = "gray"
    train(
        device=device,
        img_dir=train_img_dir,
        labels_dir=train_labels_dir,
        patch_size=patch_size,
        patch_mode=patch_mode,
        batch_size=batch_size,
        epochs_num=1000,
        max_labels=max_labels,
        model=model,
        nps_coef=0.01,
        tv_coef=2.5,
    )
    train(
        device=device,
        img_dir=train_img_dir,
        labels_dir=train_labels_dir,
        patch_size=patch_size,
        patch_mode=patch_mode,
        batch_size=batch_size,
        epochs_num=1500,
        max_labels=max_labels,
        model=model,
        nps_coef=0.01,
        tv_coef=2.5,
    )


if __name__ == '__main__':
    main()
