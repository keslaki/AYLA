import numpy as np
import matplotlib.pyplot as plt

# --- User Inputs ---
lr = float(input("Enter learning rate (e.g. 0.01): "))
epochs = int(input("Enter number of epochs (e.g. 100): "))

# --- Hyperparameters ---
beta1, beta2 = 0.9, 0.999
eps = 1e-8

# --- Data Generation ---
np.random.seed(42)
X = np.linspace(-1, 3, 100).reshape(-1, 1)
y_true = (1/3)*X**4 - (4/3)*X**3 + X**2 + (2/3)*X - (2/3)
y_true += np.random.normal(0, 0.2, y_true.shape)

# --- Activation ---
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# --- Model Functions ---
def forward(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    return Z1, A1, Z2

def mse_loss(y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)

def backward(X, y_true, Z1, A1, y_pred, W2):
    N = X.shape[0]
    dZ2 = 2 * (y_pred - y_true) / N
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0)
    return dW1, db1, dW2, db2

def adam_update(param, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

# --- Initialization ---
input_dim, hidden_dim, output_dim = 1, 128, 1
np.random.seed(42)
W1_init = np.random.randn(input_dim, hidden_dim) * 0.1
b1_init = np.zeros(hidden_dim)
W2_init = np.random.randn(hidden_dim, output_dim) * 0.1
b2_init = np.zeros(output_dim)

# --- ADAM Training ---
W1_adam, b1_adam = W1_init.copy(), b1_init.copy()
W2_adam, b2_adam = W2_init.copy(), b2_init.copy()
m_W1_adam = np.zeros_like(W1_adam)
v_W1_adam = np.zeros_like(W1_adam)
m_b1_adam = np.zeros_like(b1_adam)
v_b1_adam = np.zeros_like(b1_adam)
m_W2_adam = np.zeros_like(W2_adam)
v_W2_adam = np.zeros_like(W2_adam)
m_b2_adam = np.zeros_like(b2_adam)
v_b2_adam = np.zeros_like(b2_adam)
losses_adam = []

for epoch in range(1, epochs + 1):
    Z1, A1, y_pred = forward(X, W1_adam, b1_adam, W2_adam, b2_adam)
    loss = mse_loss(y_pred, y_true)
    losses_adam.append(loss)
    grads = backward(X, y_true, Z1, A1, y_pred, W2_adam)
    W1_adam, m_W1_adam, v_W1_adam = adam_update(W1_adam, grads[0], m_W1_adam, v_W1_adam, epoch, lr)
    b1_adam, m_b1_adam, v_b1_adam = adam_update(b1_adam, grads[1], m_b1_adam, v_b1_adam, epoch, lr)
    W2_adam, m_W2_adam, v_W2_adam = adam_update(W2_adam, grads[2], m_W2_adam, v_W2_adam, epoch, lr)
    b2_adam, m_b2_adam, v_b2_adam = adam_update(b2_adam, grads[3], m_b2_adam, v_b2_adam, epoch, lr)

print(f"Final ADAM Loss: {losses_adam[-1]:.6f}")
_, _, y_pred_adam_final = forward(X, W1_adam, b1_adam, W2_adam, b2_adam)

# --- AYLA Range Sweep ---
N1_values = np.arange(0.2, 0.9, 0.2)
N2_values = np.arange(1.0, 1.1, 0.5)
ayla_results = {}
ayla_preds = {}

for N1 in N1_values:
    for N2 in N2_values:
        W1_ayla, b1_ayla = W1_init.copy(), b1_init.copy()
        W2_ayla, b2_ayla = W2_init.copy(), b2_init.copy()
        m_W1_ayla = np.zeros_like(W1_ayla)
        v_W1_ayla = np.zeros_like(W1_ayla)
        m_b1_ayla = np.zeros_like(b1_ayla)
        v_b1_ayla = np.zeros_like(b1_ayla)
        m_W2_ayla = np.zeros_like(W2_ayla)
        v_W2_ayla = np.zeros_like(W2_ayla)
        m_b2_ayla = np.zeros_like(b2_ayla)
        v_b2_ayla = np.zeros_like(b2_ayla)
        losses_ayla = []

        for epoch in range(1, epochs + 1):
            Z1, A1, y_pred = forward(X, W1_ayla, b1_ayla, W2_ayla, b2_ayla)
            loss = mse_loss(y_pred, y_true)
            losses_ayla.append(loss)
            grads = backward(X, y_true, Z1, A1, y_pred, W2_ayla)

            if np.abs(loss) > 1:
                nnp = N2
            elif np.abs(loss) < 1:
                nnp = N1
            else:
                nnp = 1.0
            factor = nnp * (np.abs(loss) ** (nnp - 1))
            scaled_grads = [factor * g for g in grads]

            W1_ayla, m_W1_ayla, v_W1_ayla = adam_update(W1_ayla, scaled_grads[0], m_W1_ayla, v_W1_ayla, epoch, lr)
            b1_ayla, m_b1_ayla, v_b1_ayla = adam_update(b1_ayla, scaled_grads[1], m_b1_ayla, v_b1_ayla, epoch, lr)
            W2_ayla, m_W2_ayla, v_W2_ayla = adam_update(W2_ayla, scaled_grads[2], m_W2_ayla, v_W2_ayla, epoch, lr)
            b2_ayla, m_b2_ayla, v_b2_ayla = adam_update(b2_ayla, scaled_grads[3], m_b2_ayla, v_b2_ayla, epoch, lr)

        print(f"Final AYLA Loss (N1={N1:.1f}, N2={N2:.1f}): {losses_ayla[-1]:.6f}")
        ayla_results[(N1, N2)] = losses_ayla
        _, _, y_pred_final = forward(X, W1_ayla, b1_ayla, W2_ayla, b2_ayla)
        ayla_preds[(N1, N2)] = y_pred_final

# --- Plot Training Loss ---
plt.figure(figsize=(10, 7))
plt.plot(losses_adam, label="ADAM", color='black', linewidth=2)
cmap = plt.colormaps.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(ayla_results))]

for idx, ((N1, N2), losses) in enumerate(ayla_results.items()):
    plt.plot(losses, label=f"AYLA N1={N1:.1f}, N2={N2:.1f}", color=colors[idx])

plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("Losscurve.png", dpi=300)

plt.show()

# --- Plot Fits ---
plt.figure(figsize=(10, 7))
plt.scatter(X, y_true, label="True Data", color='gray', alpha=0.6)
plt.plot(X, y_pred_adam_final, label="ADAM", color='black', linewidth=2)

for idx, ((N1, N2), y_pred) in enumerate(ayla_preds.items()):
    plt.plot(X, y_pred, label=f"AYLA N1={N1:.1f}, N2={N2:.1f}", color=colors[idx])

plt.title("Fitted Curve Comparison")
plt.xlabel("X")
plt.ylabel("y_pred")
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("fittedcurve.png", dpi=300)

plt.show()
