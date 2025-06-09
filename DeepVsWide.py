import numpy as np

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Network 1: Wide (1 hidden layer, 10 neurons)
class WideNetwork:
    def __init__(self):
        self.w1 = np.random.uniform(-1, 1, (2, 10))  # input to hidden
        self.b1 = np.random.uniform(-1, 1, (1, 10))
        self.w2 = np.random.uniform(-1, 1, (10, 1))  # hidden to output
        self.b2 = np.random.uniform(-1, 1, (1, 1))

    def train(self, X, y, epochs=10000, lr=0.1):
        for epoch in range(epochs):
            # Forward
            z1 = np.dot(X, self.w1) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = sigmoid(z2)

            # Backward
            error = y - a2
            d2 = error * sigmoid_derivative(a2)
            d1 = np.dot(d2, self.w2.T) * sigmoid_derivative(a1)

            self.w2 += lr * np.dot(a1.T, d2)
            self.b2 += lr * np.sum(d2, axis=0, keepdims=True)
            self.w1 += lr * np.dot(X.T, d1)
            self.b1 += lr * np.sum(d1, axis=0, keepdims=True)

            if epoch % 1000 == 0:
                print(f"Wide Epoch {epoch}, Error: {np.mean(np.abs(error)):.4f}")

        print("Wide Final Output:", np.round(a2, 3))


# Network 2: Deep (3 hidden layers, 4 neurons each)
class DeepNetwork:
    def __init__(self):
        self.w1 = np.random.uniform(-1, 1, (2, 4))
        self.b1 = np.random.uniform(-1, 1, (1, 4))
        self.w2 = np.random.uniform(-1, 1, (4, 4))
        self.b2 = np.random.uniform(-1, 1, (1, 4))
        self.w3 = np.random.uniform(-1, 1, (4, 4))
        self.b3 = np.random.uniform(-1, 1, (1, 4))
        self.w4 = np.random.uniform(-1, 1, (4, 1))
        self.b4 = np.random.uniform(-1, 1, (1, 1))

    def train(self, X, y, epochs=10000, lr=0.1):
        for epoch in range(epochs):
            # Forward
            z1 = np.dot(X, self.w1) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = sigmoid(z2)
            z3 = np.dot(a2, self.w3) + self.b3
            a3 = sigmoid(z3)
            z4 = np.dot(a3, self.w4) + self.b4
            a4 = sigmoid(z4)

            # Backward
            error = y - a4
            d4 = error * sigmoid_derivative(a4)
            d3 = np.dot(d4, self.w4.T) * sigmoid_derivative(a3)
            d2 = np.dot(d3, self.w3.T) * sigmoid_derivative(a2)
            d1 = np.dot(d2, self.w2.T) * sigmoid_derivative(a1)

            self.w4 += lr * np.dot(a3.T, d4)
            self.b4 += lr * np.sum(d4, axis=0, keepdims=True)
            self.w3 += lr * np.dot(a2.T, d3)
            self.b3 += lr * np.sum(d3, axis=0, keepdims=True)
            self.w2 += lr * np.dot(a1.T, d2)
            self.b2 += lr * np.sum(d2, axis=0, keepdims=True)
            self.w1 += lr * np.dot(X.T, d1)
            self.b1 += lr * np.sum(d1, axis=0, keepdims=True)

            if epoch % 1000 == 0:
                print(f"Deep Epoch {epoch}, Error: {np.mean(np.abs(error)):.4f}")

        print("Deep Final Output:", np.round(a4, 3))

print("Training Wide Network:")
wide = WideNetwork()
wide.train(X, y)

print("\nTraining Deep Network:")
deep = DeepNetwork()
deep.train(X, y)

