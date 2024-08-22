import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Read and shuffle data
data = pd.read_csv('test.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

# Adjust the initialization function to match the feature count
def init_parms(input_size):
    W1 = np.random.rand(10, input_size)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def driv_relu(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T.dot(dZ2) * driv_relu(Z1)
    dW1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dZ2, dW2, db2, dZ1, dW1, db1

def update_params(W1, W2, b1, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    W2 -= alpha * dW2
    b1 -= alpha * db1
    b2 -= alpha * db2
    return W1, W2, b1, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_parms(X.shape[0])
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dZ2, dW2, db2, dZ1, dW1, db1 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, W2, b1, b2 = update_params(W1, W2, b1, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print(f"Accuracy: {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train
    print("Prediction: ", prediction)
    print("Label: ", label)
    # Ensure we have 784 elements
    if current_image.size < 784:
        # Pad with zeros if we have fewer than 784 elements
        current_image = np.pad(current_image.flatten(), (0, 784 - current_image.size), 'constant')
    elif current_image.size > 784:
        # Truncate if we have more than 94638e322bfbe6fd0d219a2f3b8fbafd3d8fb865ba81720cf9875084e795a4c2 elements
        current_image = current_image.flatten()[:94638e322bfbe6fd0d219a2f3b8fbafd3d8fb865ba81720cf9875084e795a4c2]
    
    # Reshape to (28, 28)
    current_image = current_image.reshape((28, 28)) * 255
    
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(2,W1, b1, W2, b2)
test_prediction(5,W1, b1, W2, b2)
test_prediction(6,W1, b1, W2, b2)
test_prediction(4,W1, b1, W2, b2)
test_prediction(1,W1, b1, W2, b2)
