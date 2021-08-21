import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3.6, 9.3, 7.1, 4.0, 4.2, 2.8, 7.1, 9.2, 8.1, 2.9, 9.3, 4.2])
x2 = np.array([6.6, 6.3, 8.1, 4.1, 4.2, 2.9, 7.3, 7.6, 7.8, 4.9, 8.2, 3.5])

# Data collect
features = (np.array([x1, x2])).transpose()
targets = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


def predict(features, w1, b1, w2, b2):
    """
        Predict the class
        **input: **
            *features: (Numpy Matrix)
            *w1: (Numpy Matrix)
            *b1: (Numpy vector)
            *w2: (Numpy vector)
            *b2: (Numpy scalar)
        **reutrn: (Numpy vector)**
            *0 or 1
    """
    z1 = pre_activation(features, w1, b1)
    a1 = activation(z1)
    z2 = pre_activation(a1, w2, b2)
    y = activation(z2)
    return np.round(y)


def y_predict(features, w1, b1, w2, b2):
    """
        Predict the probability
        **input: **
            *features: (Numpy Matrix)
            *w1: (Numpy Matrix)
            *b1: (Numpy vector)
            *w2: (Numpy vector)
            *b2: (Numpy scalar)
        **reutrn: (Numpy vector)**
            *0 or 1
    """
    z1 = pre_activation(features, w1, b1)
    a1 = activation(z1)
    z2 = pre_activation(a1, w2, b2)
    y = activation(z2)
    return y


def cost(prediction, target):
    """
    :param prediction: Value predicted
    :param target: Target value
    :return:
    """
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
# Hidden layer
w1 = np.random.randn(2, 3)
b1 = np.zeros(3)
# Output layer
w2 = np.random.randn(3)
b2 = np.zeros(1)

""" Activation output """

prediction = y_predict(features, w1, b1, w2, b2)
print("Accuracy = %s" % np.mean(prediction == targets))
# Plot points
plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()

for epoch in range(epochs):
    if epoch % 10 == 0:
        prediction = y_predict(features, w1, b1, w2, b2)
        print("Cost = %s" % cost(prediction, targets))
    # Init gradient
    w1_gradients = np.zeros(w1.shape)
    b1_gradients = np.zeros(b1.shape)
    w2_gradients = np.zeros(w2.shape)
    b2_gradients = np.zeros(b2.shape)
    # Go through each row
    for feature, target in zip(features, targets):
        # Compute prediction
        z1 = pre_activation(feature, w1, b1)
        a1 = activation(z1)
        z2 = pre_activation(a1, w2, b2)
        y = activation(z2)
        # Compute the error term
        error_term = (y - target)
        # Compute the error term for the output layer
        error_term_output = error_term * derive_activation(z2)
        # Compute the error term for the hidden layer
        error_term_hidden = error_term_output * w2 * derive_activation(z1)
        # Update gradients
        w1_gradients += error_term_hidden * feature[:, None]
        b1_gradients += error_term_hidden
        w2_gradients += error_term_output * a1
        b2_gradients += error_term_output
    # Update variables
    w1 = w1 - (learning_rate * w1_gradients)
    b1 = b1 - (learning_rate * b1_gradients)
    w2 = w2 - (learning_rate * w2_gradients)
    b2 = b2 - (learning_rate * b2_gradients)
# Print current Accuracy
prediction = predict(features, w1, b1, w2, b2)
print("Accuracy = %s" % np.mean(prediction == targets))
# Plot points
plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()
