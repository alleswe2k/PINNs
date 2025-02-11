import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple CNN for MNIST classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x: [batch_size, 1, 28, 28]
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # [batch_size, 16, 14, 14]
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # [batch_size, 32, 7, 7]
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 32*7*7]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST stats
])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True  # Using pin_memory can speed up host to GPU copies
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True
)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with GPU timing
num_epochs = 3
print("Starting training...")
start_time = time.time()

model.train()
for epoch in range(num_epochs):
    epoch_start = time.time()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Move data to the GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print status every 100 steps
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Step [{i + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Synchronize GPU to get accurate timing and then print epoch time
    torch.cuda.synchronize()  # Ensure all GPU work is done
    epoch_end = time.time()
    print(f"Epoch {epoch + 1} finished in {epoch_end - epoch_start:.2f} seconds")

total_time = time.time() - start_time
print(f"Training complete in {total_time:.2f} seconds")
