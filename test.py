import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def postprocess_image(tensor):
    tensor = tensor * 0.5 + 0.5  # Denormalize to [0, 1]
    image = transforms.ToPILImage()(tensor)
    return image

if __name__ == "__main__":
  
    model = SRCNN()
    model.load_state_dict(torch.load('srcnn_model.pth'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    input_image_path = r'C:\Users\rkrai\OneDrive\Desktop\Python\Intel Project\dummy.jpg'
    input_image = preprocess_image(input_image_path).to(device)


    with torch.no_grad():
        output_image = model(input_image).cpu().squeeze(0)  # Remove batch dimension

    output_image = postprocess_image(output_image)

    plt.figure(figsize=(10, 5))


    plt.subplot(1, 2, 1)
    input_image_display = transforms.ToPILImage()(input_image.cpu().squeeze(0) * 0.5 + 0.5)  # Denormalize
    plt.imshow(input_image_display)
    plt.title('Input Low-Resolution Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Output High-Resolution Image')
    plt.axis('off')

    plt.show()

    output_image.save('output_image.png')
