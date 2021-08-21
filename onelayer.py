import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3.6, 9.3, 7.1, 4.0, 4.2, 2.8, 7.1, 9.2, 8.1, 2.9, 9.3, 4.2])
x2 = np.array([6.6, 6.3, 8.1, 4.1, 4.2, 2.9, 7.3, 7.6, 7.8, 4.9, 8.2, 3.5])

# Data collect
features = (np.array([x1, x2])).transpose()
targets = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


def predict(features, weights, bias):
    a = pre_activation(features, weights, bias)
    y = activation(a)
    return np.round(y)


def cost(prediction, target):
    return np.mean((prediction - target) ** 2)


def pre_activation(features, weights, bias):
    return np.dot(features, weights) + bias


def activation(z):
    f = 1 / (1 + np.exp(- z))
    return f


def derive_activation(d):
    f = activation(d) * (1 - activation(d))
    return f


""" Variables """
learning_rate = 0.1
epochs = 1000
weights = np.random.normal(size=2)
bias = 0

""" Activation output """

prediction = predict(features, weights, bias)
print("Accuracy = %s" % np.mean(prediction == targets))
# Plot points
plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()

for epoch in range(epochs):
    if epoch % 10 == 0:
        prediction = predict(features, weights, bias)
        print("Cost = %s" % cost(prediction, targets))
    # Init gradient
    weight_gradient = np.zeros(weights.shape)
    bias_gradient = 0
    # Go through each row
    for feature, target in zip(features, targets):
        z = pre_activation(feature, weights, bias)
        y = activation(z)
        weight_gradient += (y - target) * derive_activation(z) * feature
        bias_gradient += (y - target) * derive_activation(z)
    # Update variables
    weights = weights - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient

prediction = predict(features, weights, bias)
print("Accuracy = %s" % np.mean(prediction == targets))
# Plot points
plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()
