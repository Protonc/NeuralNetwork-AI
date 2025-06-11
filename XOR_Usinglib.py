import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# XOR data
inputs = torch.tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])
targets = torch.tensor([[0.],
                        [1.],
                        [1.],
                        [0.]])

# Neural Net class
class XORdeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 4)
        self.hidden2 = nn.Linear(4,4)
        self.out = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden1(x))   # Hidden Layer 1 → ReLU
        x = self.relu(self.hidden2(x))   # Hidden Layer 2 → ReLU
        x = self.sigmoid(self.out(x))  # Output Layer → Sigmoid
        return x


# Function to train model and return losses
def train_model(lr):
    model = XORdeep()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(1000):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

# Train with different learning rates
losses_lr_01 = train_model(0.1)
losses_lr_02 = train_model(0.2)
losses_lr_05 = train_model(0.5)
losses_lr_1 = train_model(1.0)
losses_lr_2 = train_model(2.0)
losses_lr_5 = train_model(5.0)
losses_lr_10 = train_model(10.0)
losses_lr_20 = train_model(20.0)
losses_lr_50 = train_model(50.0)
losses_lr_100 = train_model(100.0)


# Plot both
plt.plot(losses_lr_01, label="lr=0.1")
plt.plot(losses_lr_02, label="lr=0.2")
plt.plot(losses_lr_05, label="lr=0.5")
plt.plot(losses_lr_1, label="lr=1.0")
plt.plot(losses_lr_2, label="lr=2.0")
plt.plot(losses_lr_5, label="lr=5.0")
plt.plot(losses_lr_10, label="lr=10.0")
plt.plot(losses_lr_20, label="lr=20.0")
plt.plot(losses_lr_50, label="lr=50.0")
plt.plot(losses_lr_100, label="lr=100.0")
plt.title("Loss vs Epochs for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


        
