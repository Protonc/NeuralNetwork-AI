import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as data
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = data.MNIST(root = './data', train = True, transform=transform, download = True)
test_data = data.MNIST(root = './data', train = False, transform = transform)
train_batch = torch.utils.data.DataLoader(dataset = train_data, batch_size = 64, shuffle = True)
test_batch = torch.utils.data.DataLoader(dataset = test_data, batch_size = 64, shuffle = True)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.h1 = nn.Linear(28*28, 128)
        self.h2 = nn.Linear(128,64)
        self.out = nn.Linear(64,10)

    def forward(self, x):
        x = self.flatten(x) 
        x = F.tanh(self.h1(x))
        x = F.tanh(self.h2(x))
        return self.out(x)

iteration = 5
model = NeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
a = 0

for epoch in range(iteration):
    model.train()
    running_loss = 0.0

    for image, label in train_batch:
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()              # empty
        output = model(image)              # forward
        loss = loss_fn(output, label)      # loss
        loss.backward()                    # backPror
        optimizer.step()                   # weight update
        a += 1

        running_loss += loss.item()
        if a % 64 == 0:
            print(f"epoch: [{epoch+1}/5], Loss: {running_loss:.4f}")
            

model.eval()
correct = 0
total = 0


with torch.no_grad():
    for image, label in test_batch:
        image, label = image.to(device), label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
accuracy = 100* correct/total
print(f"Accuracy: {accuracy:.2f}%")

# Get one batch of test images
dataiter = iter(test_batch)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Move to CPU for plotting
images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

# Plot 64 images (8x8 grid)
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i in range(64):
    ax = axes[i // 8, i % 8]
    ax.imshow(images[i].squeeze(), cmap='gray')
    pred = predicted[i].item()
    actual = labels[i].item()

    if pred != actual:
        ax.set_title(f"P:{pred}\nT:{actual}\n✗", fontsize=8, color='red')
    else:
        ax.set_title(f"P:{pred}\nT:{actual}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()
