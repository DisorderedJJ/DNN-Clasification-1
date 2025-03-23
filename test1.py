import numpy as np
import nnfs
from nnfs.datasets import spiral_data as spd

nnfs.init()
np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def __init__(self):
        self.output = None

    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        return np.mean(sample_loss)


class CCE_Loss(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)


X, Y = spd(100, 3)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5, 3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[0: 5])

loss_function = CCE_Loss()
loss = loss_function.calculate(activation2.output, Y)

print(loss)

# layer2 = Layer_Dense(5, 2)
#
# layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)
