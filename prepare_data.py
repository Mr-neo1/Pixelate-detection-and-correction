import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import numpy as np

def prepare_data(image_dir, batch_size=5, device='cpu'):
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    transform_lr = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BICUBIC),  
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    dataset = ImageFolder(root=image_dir, transform=transform_hr)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    hr_images = []
    lr_images = []

    for batch in data_loader:
        hr = batch[0].to(device)

        lr = torch.stack([transform_lr(transforms.ToPILImage()(img)) for img in hr])

        hr_images.append(hr.cpu().numpy())
        lr_images.append(lr.cpu().numpy())

    hr_images = np.concatenate(hr_images, axis=0)
    lr_images = np.concatenate(lr_images, axis=0)

    return hr_images, lr_images

if __name__ == "__main__":
    image_dir = "C:/Users/rkrai/OneDrive/Desktop/Python/Intel Project/Training data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_images, lr_images = prepare_data(image_dir, batch_size=5, device=device)

    np.save('hr_images.npy', hr_images)
    np.save('lr_images.npy', lr_images)

    save_image(torch.tensor(hr_images[0]), "hr_example.png")
    save_image(torch.tensor(lr_images[0]), "lr_example.png")
