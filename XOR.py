import random
import math

random.seed(42)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def randomf():
    return random.uniform(-1.0, 1.0)

training_input = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
training_output = [0, 1, 1, 0]

lr = 0.69
iterations = 10000


wi = [[randomf(), randomf()], [randomf(), randomf()]]  
bh = [randomf(), randomf()]                            
wh = [randomf(), randomf()]                            
bo = randomf()                                         

for epoch in range(iterations):
    total_error = 0
    for x, y in zip(training_input, training_output):
        h_input = []
        h_output = []

        for i in range(2):
            weighted_sum = wi[i][0] * x[0] + wi[i][1] * x[1] + bh[i]
            h_input.append(weighted_sum)
            h_output.append(sigmoid(weighted_sum))

        out_sum = wh[0] * h_output[0] + wh[1] * h_output[1] + bo
        output = sigmoid(out_sum)

        error = y - output
        total_error += abs(error)

        d_output = error * sigmoid_derivative(output)
        d_hidden = [0, 0]
        for i in range(2):
            d_hidden[i] = d_output * wh[i] * sigmoid_derivative(h_output[i])

        for i in range(2):
            wh[i] += lr * d_output * h_output[i]
        bo += lr * d_output

        for i in range(2):
            for j in range(2):
                wi[i][j] += lr * d_hidden[i] * x[j]
            bh[i] += lr * d_hidden[i]

    if epoch % 1000 == 0:
        print(f"Iteration {epoch}, Error: {total_error:.4f}")

print("\nFinal outputs:")
for x in training_input:
    h_output = []
    for i in range(2):
        z = wi[i][0] * x[0] + wi[i][1] * x[1] + bh[i]
        h_output.append(sigmoid(z))
    out = sigmoid(wh[0] * h_output[0] + wh[1] * h_output[1] + bo)
    print(f"Input: {x}, Output: {out:.4f}")


