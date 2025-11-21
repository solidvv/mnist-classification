import os

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision.datasets import MNIST

train_mnist_data = MNIST('.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_mnist_data = MNIST('.', train=False, transform=torchvision.transforms.ToTensor(), download=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
os.makedirs('checkpoints', exist_ok=True)

train_data_loader = torch.utils.data.DataLoader(
    train_mnist_data,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_data_loader = torch.utils.data.DataLoader(
    test_mnist_data,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

random_batch = next(iter(train_data_loader))

class CNN(nn.Module):
  def __init__(self, output=10):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(32*7*7, output)


  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)

    return x


model = CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 5

best_acc = 0.0

for epoch in range(epochs):
    running_loss = 0.0
    for _batch_idx, (images, labels) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_data_loader)

    #accuracy in test set
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            y_predicted = model(batch[0].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
            real_labels.append(batch[1])

    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    test_acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()

    if test_acc > best_acc:
      best_acc = test_acc
      torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Accuracy: {test_acc:.4f} | Best Accuracy: {best_acc:.4f}")

print("Train finished. Best model has been saved successfully")
