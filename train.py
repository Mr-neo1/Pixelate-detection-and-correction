import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x)) 
        return x

hr_images = np.load('hr_images.npy')
lr_images = np.load('lr_images.npy')
hr_images = hr_images / 255.0
lr_images = lr_images / 255.0

hr_images = torch.tensor(hr_images, dtype=torch.float32)
lr_images = torch.tensor(lr_images, dtype=torch.float32)
dataset = TensorDataset(lr_images, hr_images)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for lr, hr in data_loader:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        outputs = model(lr)
        loss = criterion(outputs, hr)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}")
torch.save(model.state_dict(), 'srcnn_model.pth')
