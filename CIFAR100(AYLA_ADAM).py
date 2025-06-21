import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------- Load and Preprocess CIFAR-100 ----------------------- #
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# ----------------------- Feature Extractor (CNN) ----------------------- #
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])

cnn.trainable = False

# Extract features
train_features = cnn.predict(x_train, batch_size=256)
test_features = cnn.predict(x_test, batch_size=256)

# ----------------------- Manual FC Classifier ----------------------- #
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

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

input_size = train_features.shape[1]
hidden_size = 128
output_size = 100

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

W1_AYLA, b1_AYLA = W1.copy(), b1.copy()
W2_AYLA, b2_AYLA = W2.copy(), b2.copy()
W1_updated, b1_updated = W1.copy(), b1.copy()
W2_updated, b2_updated = W2.copy(), b2.copy()

# ----------------------- Training ----------------------- #
N1 = float(input("Enter N1 (e.g. 1.4): "))
N2 = float(input("Enter N2 (e.g. 1.6): "))
learning_rate = float(input("Enter lr (e.g. 0.01): "))

epochs = 100
batch_size = 256
num_batches = train_features.shape[0] // batch_size

history = {
    "train_loss_orig": [], "train_acc_orig": [],
    "test_loss_orig": [], "test_acc_orig": [],
    "train_loss_updated": [], "train_acc_updated": [],
    "test_loss_updated": [], "test_acc_updated": []
}

for epoch in range(epochs):
    perm = np.random.permutation(train_features.shape[0])
    x_train_shuffled = train_features[perm]
    y_train_shuffled = y_train[perm]

    epoch_loss_orig = epoch_acc_orig = 0
    epoch_loss_updated = epoch_acc_updated = 0

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = x_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        m = X_batch.shape[0]

        # Updated model
        z1_u = np.dot(X_batch, W1_AYLA) + b1_AYLA
        a1_u = relu(z1_u)
        z2_u = np.dot(a1_u, W2_AYLA) + b2_AYLA
        y_pred_u = softmax(z2_u)
        loss_u = cross_entropy_loss(y_pred_u, y_batch)
        acc_u = accuracy(y_pred_u, y_batch)
        epoch_loss_updated += loss_u
        epoch_acc_updated += acc_u

        dz2_u = (y_pred_u - y_batch) / m
        dW2_u = np.dot(a1_u.T, dz2_u)
        db2_u = np.sum(dz2_u, axis=0, keepdims=True)
        da1_u = np.dot(dz2_u, W2_updated.T)
        dz1_u = da1_u * relu_derivative(z1_u)
        dW1_u = np.dot(X_batch.T, dz1_u)
        db1_u = np.sum(dz1_u, axis=0, keepdims=True)

        # Original model
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = softmax(z2)
        loss = cross_entropy_loss(y_pred, y_batch)
        acc = accuracy(y_pred, y_batch)
        epoch_loss_orig += loss
        epoch_acc_orig += acc

        dz2 = (y_pred - y_batch) / m
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(X_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        if abs(loss_u) > 1:
            nnp = N2
        else:
            nnp = N1
        factor = nnp * (abs(loss_u)) ** (nnp - 1)
       
        
        # Ayla model
        z1_AYLA = np.dot(X_batch, W1_AYLA) + b1_AYLA
        a1_AYLA = relu(z1_AYLA)
        z2_AYLA = np.dot(a1_AYLA, W2_AYLA) + b2_AYLA
        y_pred_AYLA = softmax(z2_AYLA)        
        dz2_AYLA = factor * (y_pred_AYLA - y_batch) / m
        dW2_AYLA = np.dot(a1_u.T, dz2_AYLA)
        db2_AYLA = np.sum(dz2_AYLA, axis=0, keepdims=True)
        da1_AYLA = np.dot(dz2_AYLA, W2_AYLA.T)
        dz1_AYLA = da1_AYLA * relu_derivative(z1_AYLA)
        dW1_AYLA = np.dot(X_batch.T, dz1_AYLA)
        db1_AYLA = np.sum(dz1_AYLA, axis=0, keepdims=True)
        
        
        
        print(loss , loss_u)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1_AYLA -= learning_rate * dW1_AYLA
        b1_AYLA -= learning_rate * db1_AYLA
        W2_AYLA -= learning_rate * dW2_AYLA
        b2_AYLA -= learning_rate * db2_AYLA

        W1_updated -= learning_rate * dW1_u
        b1_updated -= learning_rate * db1_u
        W2_updated -= learning_rate * dW2_u
        b2_updated -= learning_rate * db2_u

    def eval_model(X, Y, W1, b1, W2, b2):
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = softmax(z2)
        return cross_entropy_loss(y_pred, Y), accuracy(y_pred, Y)

    loss_test_orig, acc_test_orig = eval_model(test_features, y_test, W1, b1, W2, b2)
    loss_test_updated, acc_test_updated = eval_model(test_features, y_test, W1_AYLA, b1_AYLA, W2_AYLA, b2_AYLA)

    history["train_loss_orig"].append(epoch_loss_orig / num_batches)
    history["train_acc_orig"].append(epoch_acc_orig / num_batches)
    history["test_loss_orig"].append(loss_test_orig)
    history["test_acc_orig"].append(acc_test_orig)
    history["train_loss_updated"].append(epoch_loss_updated / num_batches)
    history["train_acc_updated"].append(epoch_acc_updated / num_batches)
    history["test_loss_updated"].append(loss_test_updated)
    history["test_acc_updated"].append(acc_test_updated)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f" Original - Train Loss: {epoch_loss_orig / num_batches:.4f}, Train Acc: {epoch_acc_orig / num_batches:.4f}, Test Loss: {loss_test_orig:.4f}, Test Acc: {acc_test_orig:.4f}")
    print(f" AYLA      - Train Loss: {epoch_loss_updated / num_batches:.4f}, Train Acc: {epoch_acc_updated / num_batches:.4f}, Test Loss: {loss_test_updated:.4f}, Test Acc: {acc_test_updated:.4f}")
    print("-" * 70)

# ----------------------- Plot Results ----------------------- #
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs_range, history["train_loss_orig"], label="Train Loss Original", color='black')
plt.plot(epochs_range, history["test_loss_orig"], label="Test Loss Original", linestyle='--', color='black')
plt.plot(epochs_range, history["train_loss_updated"], label="Train Loss AYLA", color='blue')
plt.plot(epochs_range, history["test_loss_updated"], label="Test Loss AYLA", linestyle='--', color='blue')
plt.grid(linestyle='dotted')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.savefig("Loss3.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(epochs_range, history["train_acc_orig"], label="Train Acc Original", color='black')
plt.plot(epochs_range, history["test_acc_orig"], label="Test Acc Original", linestyle='--', color='black')
plt.plot(epochs_range, history["train_acc_updated"], label="Train Acc AYLA", color='blue')
plt.plot(epochs_range, history["test_acc_updated"], label="Test Acc AYLA", linestyle='--', color='blue')
plt.grid(linestyle='dotted')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.savefig("Accu3.png", dpi=300)
plt.show()
