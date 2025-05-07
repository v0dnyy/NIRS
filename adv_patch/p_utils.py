import numpy as np
import torch
from matplotlib import pyplot as plt

from my_utils import to_array


def init_patch(patch_type, image_size, noise_percentage):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
        return patch


def mask_generation(mask_type, patch, image_size):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)
        x_location, y_location = np.random.randint(low=0, high=image_size[1] - patch.shape[1]), np.random.randint(low=0,
                                                                                                                  high=image_size[2] -patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
        mask = applied_patch.copy()
        mask[mask != 0] = 1.0
        return applied_patch, mask, x_location, y_location


def test_patch(device, patch_type, target_label, patch, test_loader, model, labels):
    model.eval()
    total, success = 0, 0
    for (image, label) in test_loader:
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data != target_label:
            continue
        total += 1
        applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image.shape[1:])
        applied_patch = torch.from_numpy(applied_patch).float().to(device)
        mask = torch.from_numpy(mask).float().to(device)
        image_squeezed = image.squeeze(0).float().to(device)
        perturbated_image = mask * applied_patch + (1 - mask) * image_squeezed
        output = model(perturbated_image.float().unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        plt.imshow(to_array(perturbated_image))
        plt.title(labels[predicted[0].data])
        plt.show()
        if predicted[0].data == target_label:
            success += 1
    return total / success if success > 0 else 0