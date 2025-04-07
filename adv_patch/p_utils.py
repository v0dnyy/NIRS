import numpy as np


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
                                                                                                                  high=
                                                                                                                  image_size[
                                                                                                                      2] -
                                                                                                                  patch.shape[
                                                                                                                      2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
        mask = applied_patch.copy()
        mask[mask != 0] = 1.0
        return applied_patch, mask, x_location, y_location
