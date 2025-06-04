import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
import random

# 🧠 Brain Definition
class MiniBrain(nn.Module):
    def __init__(self):
        super(MiniBrain, self).__init__()
        self.fc1 = nn.Linear(4, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# 🎮 Simulation + GUI
class MiniBrainApp:
    def __init__(self, master):
        self.master = master
        master.title("MiniBrain Simulation")

        self.brain = MiniBrain()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

        self.canvas = tk.Canvas(master, width=400, height=300, bg="white")
        self.canvas.pack()

        self.info = tk.Label(master, text="", font=("Consolas", 12))
        self.info.pack()

        self.step_button = tk.Button(master, text="🧠 Step", command=self.step)
        self.step_button.pack(pady=5)

        self.epoch = 0

    def step(self):
        self.epoch += 1

        # Random input as world state
        input_data = torch.rand(1, 4)

        # Brain decides
        output = self.brain(input_data)
        action = torch.argmax(output).item()

        # Simulate reward: reward if action == 1
        reward = 1.0 if action == 1 else 0.0
        target = torch.tensor([[1.0 - reward, reward]])

        # Training step
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Display input as colors
        self.canvas.delete("all")
        colors = ['#00f', '#0f0', '#f00', '#ff0']
        for i, val in enumerate(input_data[0]):
            self.canvas.create_rectangle(i * 100, 0, (i+1) * 100, 100, fill=colors[i])
            self.canvas.create_text(i * 100 + 50, 50, text=f"{val.item():.2f}", font=("Arial", 12), fill="white")

        # Show output
        self.info.config(text=(
            f"Epoch: {self.epoch}\n"
            f"Input: {[round(v.item(), 2) for v in input_data[0]]}\n"
            f"Output: {[round(v.item(), 2) for v in output[0]]} (Action {action})\n"
            f"Reward: {reward}"
        ))

# 🧠 Launch GUI
root = tk.Tk()
app = MiniBrainApp(root)
root.mainloop()
