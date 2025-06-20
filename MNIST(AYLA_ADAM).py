import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Preprocess inputs
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Add Gaussian noise
noise_level = 0.7  # you can change this value
np.random.seed(42)  # reproducibility
x_train += np.random.normal(0, noise_level, x_train.shape)
x_test += np.random.normal(0, noise_level, x_test.shape)

# Clip values to stay within [0, 1] after adding noise
x_train = np.clip(x_train, 0.0, 1.0)
x_test = np.clip(x_test, 0.0, 1.0)


def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Network architecture
input_size = 784
hidden_size = 128
output_size = 10


# Standard manual initialization
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


# Create exact copy for AYLA network
W1_AYLA = W1.copy()
b1_AYLA = b1.copy()
W2_AYLA = W2.copy()
b2_AYLA = b2.copy()

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    clipped_preds = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(clipped_preds)) / m

def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)

# ADAM hyperparameters
learning_rate = 1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize ADAM moments for original network
m_W1 = np.zeros_like(W1)
v_W1 = np.zeros_like(W1)
m_b1 = np.zeros_like(b1)
v_b1 = np.zeros_like(b1)

m_W2 = np.zeros_like(W2)
v_W2 = np.zeros_like(W2)
m_b2 = np.zeros_like(b2)
v_b2 = np.zeros_like(b2)

# Initialize ADAM moments for AYLA network
m_W1_AYLA = np.zeros_like(W1_AYLA)
v_W1_AYLA = np.zeros_like(W1_AYLA)
m_b1_AYLA = np.zeros_like(b1_AYLA)
v_b1_AYLA = np.zeros_like(b1_AYLA)

m_W2_AYLA = np.zeros_like(W2_AYLA)
v_W2_AYLA = np.zeros_like(W2_AYLA)
m_b2_AYLA = np.zeros_like(b2_AYLA)
v_b2_AYLA = np.zeros_like(b2_AYLA)

# User inputs for N1, N2
N1 = float(input("Enter N1 (e.g. 1.4): "))
N2 = float(input("Enter N2 (e.g. 1.6): "))

epochs = 50
batch_size = 256
num_batches = x_train.shape[0] // batch_size

def adam_update(param, grad, m_param, v_param, t):
    m_param = beta1 * m_param + (1 - beta1) * grad
    v_param = beta2 * v_param + (1 - beta2) * (grad ** 2)
    m_hat = m_param / (1 - beta1 ** t)
    v_hat = v_param / (1 - beta2 ** t)
    param_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    param -= param_update
    return param, m_param, v_param

# Store metrics for plotting
history = {
    "train_loss_orig": [], "train_acc_orig": [],
    "test_loss_orig": [], "test_acc_orig": [],
    "train_loss_AYLA": [], "train_acc_AYLA": [],
    "test_loss_AYLA": [], "test_acc_AYLA": [],
}

for epoch in range(epochs):
    perm = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train_oh[perm]
    
    epoch_loss_orig = 0
    epoch_acc_orig = 0
    epoch_loss_AYLA = 0
    epoch_acc_AYLA = 0
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = x_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        m = batch_size
        
        # Forward original
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = softmax(z2)
        loss = cross_entropy_loss(y_pred, y_batch)
 
        acc = accuracy(y_pred, y_batch)
        epoch_loss_orig += loss
        epoch_acc_orig += acc
        
        # Backprop original
        dz2 = (y_pred - y_batch) / m
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Forward AYLA
        z1_AYLA = np.dot(X_batch, W1_AYLA) + b1_AYLA
        a1_AYLA = relu(z1_AYLA)
        z2_AYLA = np.dot(a1_AYLA, W2_AYLA) + b2_AYLA
        y_pred_AYLA = softmax(z2_AYLA)
        loss_AYLA = cross_entropy_loss(y_pred_AYLA, y_batch)
        acc_AYLA = accuracy(y_pred_AYLA, y_batch)
        epoch_loss_AYLA += loss_AYLA
        epoch_acc_AYLA += acc_AYLA
        
        # Backprop AYLA with your condition applied
        
        # Condition and factor calculation based on original loss
        condition = abs(loss) > 1
        condition2 = abs(loss) < 1
        
        # Compute nnp scalar
        if condition:
            nnp = N2
        elif condition2:
            nnp = N1
        else:
            nnp = 1
        
        factor = nnp * (abs(loss_AYLA)) ** (nnp - 1)
        
        dz2_AYLA = factor * (y_pred_AYLA - y_batch) / m
        
        dW2_AYLA = np.dot(a1_AYLA.T, dz2_AYLA)
        db2_AYLA = np.sum(dz2_AYLA, axis=0, keepdims=True)
        da1_AYLA = np.dot(dz2_AYLA, W2_AYLA.T)
        dz1_AYLA = da1_AYLA * relu_derivative(z1_AYLA)
        dW1_AYLA = np.dot(X_batch.T, dz1_AYLA)
        db1_AYLA = np.sum(dz1_AYLA, axis=0, keepdims=True)
        
        t = epoch * num_batches + i + 1
        
        # Update original
        W1, m_W1, v_W1 = adam_update(W1, dW1, m_W1, v_W1, t)
        b1, m_b1, v_b1 = adam_update(b1, db1, m_b1, v_b1, t)
        W2, m_W2, v_W2 = adam_update(W2, dW2, m_W2, v_W2, t)
        b2, m_b2, v_b2 = adam_update(b2, db2, m_b2, v_b2, t)
        
        # Update AYLA
        W1_AYLA, m_W1_AYLA, v_W1_AYLA = adam_update(W1_AYLA, dW1_AYLA, m_W1_AYLA, v_W1_AYLA, t)
        b1_AYLA, m_b1_AYLA, v_b1_AYLA = adam_update(b1_AYLA, db1_AYLA, m_b1_AYLA, v_b1_AYLA, t)
        W2_AYLA, m_W2_AYLA, v_W2_AYLA = adam_update(W2_AYLA, dW2_AYLA, m_W2_AYLA, v_W2_AYLA, t)
        b2_AYLA, m_b2_AYLA, v_b2_AYLA = adam_update(b2_AYLA, db2_AYLA, m_b2_AYLA, v_b2_AYLA, t)
    
    epoch_loss_orig /= num_batches
    epoch_acc_orig /= num_batches
    epoch_loss_AYLA /= num_batches
    epoch_acc_AYLA /= num_batches
    
    # Evaluate test sets
    def eval_test(W1_, b1_, W2_, b2_):
        z1_test = np.dot(x_test, W1_) + b1_
        a1_test = relu(z1_test)
        z2_test = np.dot(a1_test, W2_) + b2_
        y_test_pred = softmax(z2_test)
        loss_test = cross_entropy_loss(y_test_pred, y_test_oh)
        acc_test = accuracy(y_test_pred, y_test_oh)
        return loss_test, acc_test
    
    loss_test_orig, acc_test_orig = eval_test(W1, b1, W2, b2)
    loss_test_AYLA, acc_test_AYLA = eval_test(W1_AYLA, b1_AYLA, W2_AYLA, b2_AYLA)
    
    history["train_loss_orig"].append(epoch_loss_orig)
    history["train_acc_orig"].append(epoch_acc_orig)
    history["test_loss_orig"].append(loss_test_orig)
    history["test_acc_orig"].append(acc_test_orig)
    
    history["train_loss_AYLA"].append(epoch_loss_AYLA)
    history["train_acc_AYLA"].append(epoch_acc_AYLA)
    history["test_loss_AYLA"].append(loss_test_AYLA)
    history["test_acc_AYLA"].append(acc_test_AYLA)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f" Original - Train Loss: {epoch_loss_orig:.4f}, Train Acc: {epoch_acc_orig:.4f}, Test Loss: {loss_test_orig:.4f}, Test Acc: {acc_test_orig:.4f}")
    print(f" AYLA      - Train Loss: {epoch_loss_AYLA:.4f}, Train Acc: {epoch_acc_AYLA:.4f}, Test Loss: {loss_test_AYLA:.4f}, Test Acc: {acc_test_AYLA:.4f}")
    print("-" * 70)



# Plot results
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs_range, history["train_loss_orig"], label="Train Loss Original", color='black')
plt.plot(epochs_range, history["test_loss_orig"], label="Test Loss Original", linestyle='--', color='black')
plt.plot(epochs_range, history["train_loss_AYLA"], label="Train Loss AYLA", color='blue')
plt.plot(epochs_range, history["test_loss_AYLA"], label="Test Loss AYLA", linestyle='--', color='blue')
plt.grid(linestyle='dotted')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("Loss.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))

plt.plot(epochs_range, history["train_acc_orig"], label="Train Acc Original", color='black')
plt.plot(epochs_range, history["test_acc_orig"], label="Test Acc Original", linestyle='--', color='black')
plt.plot(epochs_range, history["train_acc_AYLA"], label="Train Acc AYLA", color='blue')
plt.plot(epochs_range, history["test_acc_AYLA"], label="Test Acc AYLA", linestyle='--', color='blue')
plt.grid(linestyle='dotted')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.savefig("Accu.png", dpi=300)
plt.show()
