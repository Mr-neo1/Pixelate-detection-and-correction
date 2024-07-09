import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)  # Upsample to match HR size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


hr_images = np.load('hr_images.npy')
lr_images = np.load('lr_images.npy')
lr_images = torch.tensor(lr_images, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
model.load_state_dict(torch.load('srcnn_model.pth'))
model.eval()

with torch.no_grad():
    lr_images = lr_images.to(device)
    sr_images = model(lr_images).cpu().numpy()

def display_images(original, low_res, super_res):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].imshow((original.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))
    axs[0].set_title('Original High-Resolution')
    axs[0].axis('off')

    axs[1].imshow((low_res.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))
    axs[1].set_title('Low-Resolution')
    axs[1].axis('off')

    axs[2].imshow((super_res.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))
    axs[2].set_title('Super-Resolution')
    axs[2].axis('off')
    plt.show()
display_images(hr_images[0], lr_images[0].cpu().numpy(), sr_images[0])
