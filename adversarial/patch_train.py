from torch import optim
from tqdm import tqdm
import torch
import torchvision
import patch_utils
from dataset import PersonDataset
import torch


def train(device, img_dir, lab_dir, batch_size, epochs_num, max_labels, model):
    start_learning_rate = 0.03
    adv_patch = patch_utils.generate_patch(300, device).requires_grad_(True)
    train_loader = torch.utils.data.DataLoader(PersonDataset(img_dir, lab_dir, max_labels, 640,
                                                             shuffle=True),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    epoch_length = len(train_loader)
    print(f'One epoch is {len(train_loader)}')
    scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
    optimizer = optim.Adam([adv_patch], lr=start_learning_rate, amsgrad=True)
    scheduler = scheduler_factory(optimizer)
    for epoch in range(epochs_num):
        ep_detection_loss = 0
        ep_total_variation_loss = 0
        ep_loss = 0
        for i_batch, (img_batch, label_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            # with torch.autograd.detect_anomaly():
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
                adv_patch = adv_patch.to(device)
                adv_batch_transformed = patch_utils.transform_patch(device, adv_patch, label_batch, 640, do_rotate=True, rand_loc=False)
                p_img_batch = patch_utils.apply_patch_to_img_batch(img_batch, adv_batch_transformed)

                # img = p_img_batch[1, :, :]
                # img = torchvision.transforms.ToPILImage('RGB')(img.detach().cpu())
                # img.show()

                raw_outputs = model(p_img_batch)
                max_prob = patch_utils.max_prob_extraction(raw_outputs[1], 0, 80)
                total_variation = patch_utils.calc_total_variation(adv_patch)
                total_variation_loss = total_variation * 2.5
                detection_loss = max_prob.mean()
                loss = detection_loss + torch.max(total_variation_loss, torch.tensor(0.1).to(device))
                ep_detection_loss += detection_loss.detach().cpu().numpy()
                ep_total_variation_loss += total_variation_loss.detach().cpu().numpy()
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch.data.clamp_(0, 1)

                if i_batch + 1 >= len(train_loader):
                    pass
                else:
                    del adv_batch_transformed, p_img_batch, raw_outputs, max_prob, detection_loss, total_variation_loss, loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        ep_detection_loss = ep_detection_loss / len(train_loader)
        ep_total_variation_loss = ep_total_variation_loss / len(train_loader)
        ep_loss = ep_loss / len(train_loader)

    img = torchvision.transforms.ToPILImage('RGB')(adv_patch)
    img.show()
    img.save(f"../adversarial/patch_e_{epochs_num}_b_{batch_size}.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False).eval()
    train_img_dir = '../adversarial/dataset/train/images'
    train_labels_dir = '../adversarial/dataset/train/labels'
    batch_size = 8
    epochs_num = 1
    max_labels = 24
    train(device, train_img_dir, train_labels_dir, batch_size, epochs_num, max_labels, model)


if __name__ == '__main__':
    main()
