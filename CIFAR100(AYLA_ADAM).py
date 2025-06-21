
# ----------------------- Architecture & Initialization ----------------------- #
input_size = 3072  # 32*32*3
hidden_size = 128
output_size = 100  # CIFAR100 has 100 classes

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# AYLA + Updated copies
W1_AYLA, b1_AYLA = W1.copy(), b1.copy()
W2_AYLA, b2_AYLA = W2.copy(), b2.copy()

W1_updated, b1_updated = W1.copy(), b1.copy()
W2_updated, b2_updated = W2.copy(), b2.copy()

# ----------------------- Activation and Utility Functions ----------------------- #
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

# ----------------------- Optimizer Setup ----------------------- #
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

def adam_update(param, grad, m_param, v_param, t):
    m_param = beta1 * m_param + (1 - beta1) * grad
    v_param = beta2 * v_param + (1 - beta2) * (grad ** 2)
    m_hat = m_param / (1 - beta1 ** t)
    v_hat = v_param / (1 - beta2 ** t)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m_param, v_param

# Initialize ADAM moments for all networks
def init_adam_moments(W, b):
    return np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)

m_W1, v_W1, m_b1, v_b1 = init_adam_moments(W1, b1)
m_W2, v_W2, m_b2, v_b2 = init_adam_moments(W2, b2)

m_W1_AYLA, v_W1_AYLA, m_b1_AYLA, v_b1_AYLA = init_adam_moments(W1_AYLA, b1_AYLA)
m_W2_AYLA, v_W2_AYLA, m_b2_AYLA, v_b2_AYLA = init_adam_moments(W2_AYLA, b2_AYLA)

m_W1_updated, v_W1_updated, m_b1_updated, v_b1_updated = init_adam_moments(W1_updated, b1_updated)
m_W2_updated, v_W2_updated, m_b2_updated, v_b2_updated = init_adam_moments(W2_updated, b2_updated)

# ----------------------- Hyperparameters ----------------------- #
N1 = float(input("Enter N1 (e.g. 1.4): "))
N2 = float(input("Enter N2 (e.g. 1.6): "))
learning_rate = float(input("Enter lr (e.g. 0.01): "))

epochs = 40
batch_size = 256
num_batches = x_train.shape[0] // batch_size

# ----------------------- Training Loop ----------------------- #
history = {
    "train_loss_orig": [], "train_acc_orig": [],
    "test_loss_orig": [], "test_acc_orig": [],
    "train_loss_updated": [], "train_acc_updated": [],
    "test_loss_updated": [], "test_acc_updated": [],
}

for epoch in range(epochs):
    perm = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train_oh[perm]

    epoch_loss_orig = epoch_acc_orig = 0
    epoch_loss_AYLA = epoch_acc_AYLA = 0
    epoch_loss_updated = epoch_acc_updated = 0

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = x_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        m = batch_size

        # ---- Forward & Backprop: Updated (using AYLA weights) ---- #
        z1_updated = np.dot(X_batch, W1_AYLA) + b1_AYLA
        a1_updated = relu(z1_updated)
        z2_updated = np.dot(a1_updated, W2_AYLA) + b2_AYLA
        y_pred_updated = softmax(z2_updated)

        loss_updated = cross_entropy_loss(y_pred_updated, y_batch)
        acc_updated = accuracy(y_pred_updated, y_batch)
        epoch_loss_updated += loss_updated
        epoch_acc_updated += acc_updated

        dz2_updated = (y_pred_updated - y_batch) / m
        dW2_updated = np.dot(a1_updated.T, dz2_updated)
        db2_updated = np.sum(dz2_updated, axis=0, keepdims=True)
        da1_updated = np.dot(dz2_updated, W2_updated.T)
        dz1_updated = da1_updated * relu_derivative(z1_updated)
        dW1_updated = np.dot(X_batch.T, dz1_updated)
        db1_updated = np.sum(dz1_updated, axis=0, keepdims=True)

        # ---- Forward & Backprop: Original ---- #
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

        # ---- Forward & Backprop: AYLA ---- #
        z1_AYLA = np.dot(X_batch, W1_AYLA) + b1_AYLA
        a1_AYLA = relu(z1_AYLA)
        z2_AYLA = np.dot(a1_AYLA, W2_AYLA) + b2_AYLA
        y_pred_AYLA = softmax(z2_AYLA)

        loss_AYLA = cross_entropy_loss(y_pred_AYLA, y_batch)
        acc_AYLA = accuracy(y_pred_AYLA, y_batch)
        epoch_loss_AYLA += loss_AYLA
        epoch_acc_AYLA += acc_AYLA

        # AYLA loss modifier
        if abs(loss_updated) > 1:
            nnp = N2
        else:
            nnp = N1
        factor = nnp * (abs(loss_updated)) ** (nnp - 1)

        dz2_AYLA = factor * (y_pred_AYLA - y_batch) / m
        dW2_AYLA = np.dot(a1_AYLA.T, dz2_AYLA)
        db2_AYLA = np.sum(dz2_AYLA, axis=0, keepdims=True)
        da1_AYLA = np.dot(dz2_AYLA, W2_AYLA.T)
        dz1_AYLA = da1_AYLA * relu_derivative(z1_AYLA)
        dW1_AYLA = np.dot(X_batch.T, dz1_AYLA)
        db1_AYLA = np.sum(dz1_AYLA, axis=0, keepdims=True)

        # ---- Update: Original ---- #
        t = epoch * num_batches + i + 1
        W1, m_W1, v_W1 = adam_update(W1, dW1, m_W1, v_W1, t)
        b1, m_b1, v_b1 = adam_update(b1, db1, m_b1, v_b1, t)
        W2, m_W2, v_W2 = adam_update(W2, dW2, m_W2, v_W2, t)
        b2, m_b2, v_b2 = adam_update(b2, db2, m_b2, v_b2, t)

        # ---- Update: AYLA ---- #
        W1_AYLA, m_W1_AYLA, v_W1_AYLA = adam_update(W1_AYLA, dW1_AYLA, m_W1_AYLA, v_W1_AYLA, t)
        b1_AYLA, m_b1_AYLA, v_b1_AYLA = adam_update(b1_AYLA, db1_AYLA, m_b1_AYLA, v_b1_AYLA, t)
        W2_AYLA, m_W2_AYLA, v_W2_AYLA = adam_update(W2_AYLA, dW2_AYLA, m_W2_AYLA, v_W2_AYLA, t)
        b2_AYLA, m_b2_AYLA, v_b2_AYLA = adam_update(b2_AYLA, db2_AYLA, m_b2_AYLA, v_b2_AYLA, t)

        # ---- Copy Weights & Update: Updated ---- #
        W1_updated, b1_updated = W1_AYLA.copy(), b1_AYLA.copy()
        W2_updated, b2_updated = W2_AYLA.copy(), b2_AYLA.copy()

        W1_updated, m_W1_updated, v_W1_updated = adam_update(W1_updated, dW1_updated, m_W1_updated, v_W1_updated, t)
        b1_updated, m_b1_updated, v_b1_updated = adam_update(b1_updated, db1_updated, m_b1_updated, v_b1_updated, t)
        W2_updated, m_W2_updated, v_W2_updated = adam_update(W2_updated, dW2_updated, m_W2_updated, v_W2_updated, t)
        b2_updated, m_b2_updated, v_b2_updated = adam_update(b2_updated, db2_updated, m_b2_updated, v_b2_updated, t)

    # ----------------------- Epoch Summary ----------------------- #
    epoch_loss_orig /= num_batches
    epoch_acc_orig /= num_batches
    epoch_loss_updated /= num_batches
    epoch_acc_updated /= num_batches

    def eval_test(W1_, b1_, W2_, b2_):
        z1_test = np.dot(x_test, W1_) + b1_
        a1_test = relu(z1_test)
        z2_test = np.dot(a1_test, W2_) + b2_
        y_test_pred = softmax(z2_test)
        loss_test = cross_entropy_loss(y_test_pred, y_test_oh)
        acc_test = accuracy(y_test_pred, y_test_oh)
        return loss_test, acc_test

    loss_test_orig, acc_test_orig = eval_test(W1, b1, W2, b2)
    loss_test_updated, acc_test_updated = eval_test(W1_AYLA, b1_AYLA, W2_AYLA, b2_AYLA)

    history["train_loss_orig"].append(epoch_loss_orig)
    history["train_acc_orig"].append(epoch_acc_orig)
    history["test_loss_orig"].append(loss_test_orig)
    history["test_acc_orig"].append(acc_test_orig)

    history["train_loss_updated"].append(epoch_loss_updated)
    history["train_acc_updated"].append(epoch_acc_updated)
    history["test_loss_updated"].append(loss_test_updated)
    history["test_acc_updated"].append(acc_test_updated)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f" Original - Train Loss: {epoch_loss_orig:.4f}, Train Acc: {epoch_acc_orig:.4f}, Test Loss: {loss_test_orig:.4f}, Test Acc: {acc_test_orig:.4f}")
    print(f" AYLA      - Train Loss: {epoch_loss_updated:.4f}, Train Acc: {epoch_acc_updated:.4f}, Test Loss: {loss_test_updated:.4f}, Test Acc: {acc_test_updated:.4f}")
    print("-" * 70)

# Plot results
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs_range, history["train_loss_orig"], label="Train Loss Original", color='black')
plt.plot(epochs_range, history["test_loss_orig"], label="Test Loss Original", linestyle='--', color='black')
plt.plot(epochs_range, history["train_loss_updated"], label="Train Loss AYLA", color='blue')
plt.plot(epochs_range, history["test_loss_updated"], label="Test Loss AYLA", linestyle='--', color='blue')
plt.grid(linestyle='dotted')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(r"Loss ($N_1$ = {:.2f}, $N_2$ = {:.8f})".format(N1, N2))
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
plt.title(r"Accuracy ($N_1$ = {:.2f}, $N_2$ = {:.10f})".format(N1, N2))

plt.legend()
plt.savefig("Accu3.png", dpi=300)
plt.show()
