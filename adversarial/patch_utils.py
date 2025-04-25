import torch

def generate_patch(patch_size, device):
    return torch.rand((3, patch_size, patch_size)).to(device)




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_img = generate_patch(32, device)
    pass


if __name__ == '__main__':
    main()