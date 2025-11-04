import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),        
])


train_dataset = datasets.ImageFolder(root='/home/tjx/ManiSkill-ViTac2024-main/image_data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)  
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = len(train_dataset.classes) 
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(images) 
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), '/home/tjx/ManiSkill-ViTac2024-main/task_classification_model/classification.pth')


def predict(image_path):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return train_dataset.classes[predicted.item()]
