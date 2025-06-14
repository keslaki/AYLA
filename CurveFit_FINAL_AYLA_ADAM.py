import numpy as np
import matplotlib.pyplot as plt

# ---------- Generate Data ----------
np.random.seed(42)

numerbss, randomm = 100, 0.2
X_ayla = np.linspace(-1, 3, numerbss).reshape(-1, 1)
y_true = (1/3)*X_ayla**4 - (4/3)*X_ayla**3 + X_ayla**2 + (2/3)*X_ayla - (2/3)
y_true += np.random.normal(0, randomm, (numerbss, 1))

X_input = X_ayla
y_input = y_true

# Hyperparameters
epochs = 200
lr = 0.01
beta1, beta2 = 0.9, 0.999
eps = 1e-8

N1, N2 = 1, .15

# Network architecture
input_dim = 1
hidden_dim = 100
output_dim = 1

# ---------- Initialize parameters ----------
def init_params():
    W1 = np.random.randn(input_dim, hidden_dim) * 0.5
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.5
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

W1_adam, b1_adam, W2_adam, b2_adam = init_params()
W1_ayla, b1_ayla, W2_ayla, b2_ayla = W1_adam.copy(), b1_adam.copy(), W2_adam.copy(), b2_adam.copy()

# ADAM state variables
mw1, vw1 = np.zeros_like(W1_adam), np.zeros_like(W1_adam)
mb1, vb1 = np.zeros_like(b1_adam), np.zeros_like(b1_adam)
mw2, vw2 = np.zeros_like(W2_adam), np.zeros_like(W2_adam)
mb2, vb2 = np.zeros_like(b2_adam), np.zeros_like(b2_adam)

mw1_ayla, vw1_ayla = np.zeros_like(W1_adam), np.zeros_like(W1_adam)
mb1_ayla, vb1_ayla = np.zeros_like(b1_adam), np.zeros_like(b1_adam)
mw2_ayla, vw2_ayla = np.zeros_like(W2_adam), np.zeros_like(W2_adam)
mb2_ayla, vb2_ayla = np.zeros_like(b2_adam), np.zeros_like(b2_adam)

adam_losses, ayla_losses = [], []

# ---------- Training Loop ----------
for epoch in range(1, epochs+1):

    # Forward pass (shared for both)
    z1 = X_input @ W1_adam + b1_adam  # shape: (100, hidden_dim)
    a1 = np.tanh(z1)
    y_pred = a1 @ W2_adam + b2_adam
    loss = y_pred - y_input
    mse = np.mean(loss**2)
    adam_losses.append(mse)

    # Backpropagation
    dL = 2 * loss / numerbss

    dW2 = a1.T @ dL
    db2 = np.sum(dL, axis=0, keepdims=True)

    da1 = dL @ W2_adam.T
    dz1 = da1 * (1 - a1**2)

    dW1 = X_input.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ADAM updates
    for (grad, m, v, param) in zip([dW1, db1, dW2, db2],
                                   [mw1, mb1, mw2, mb2],
                                   [vw1, vb1, vw2, vb2],
                                   [W1_adam, b1_adam, W2_adam, b2_adam]):
        m *= beta1
        m += (1-beta1)*grad
        v *= beta2
        v += (1-beta2)*(grad**2)
        m_hat = m / (1 - beta1**epoch)
        v_hat = v / (1 - beta2**epoch)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # --- AYLA-ADAM ---
    z1_ayla = X_input @ W1_ayla + b1_ayla
    a1_ayla = np.tanh(z1_ayla)
    y_pred_ayla = a1_ayla @ W2_ayla + b2_ayla
    
    

    # Backpropagation
    
    loss_ayla = y_pred_ayla - y_input
    mse_ayla = np.mean(loss_ayla**2)
    ayla_losses.append(mse_ayla)

    abs_loss = np.abs(mse_ayla)
    condition1 = abs_loss > 1
    condition2 = abs_loss < 1
    nnp = np.where(condition1, N2, np.where(condition2, N1, 1))
    factorr = nnp * abs_loss ** (nnp - 1)

    grad_loss = 2 * factorr *  loss_ayla / numerbss
    dW2_ayla = a1_ayla.T @ grad_loss
    db2_ayla = np.sum(grad_loss, axis=0, keepdims=True)

    da1_ayla = grad_loss @ W2_ayla.T
    dz1_ayla = da1_ayla * (1 - a1_ayla**2)

    dW1_ayla = X_input.T @ dz1_ayla
    db1_ayla = np.sum(dz1_ayla, axis=0, keepdims=True)

    for (grad_ayla, m_ayla, v_ayla, param_ayla) in zip([dW1_ayla, db1_ayla, dW2_ayla, db2_ayla],
                                   [mw1_ayla, mb1_ayla, mw2_ayla, mb2_ayla],
                                   [vw1_ayla, vb1_ayla, vw2_ayla, vb2_ayla],
                                   [W1_ayla, b1_ayla, W2_ayla, b2_ayla]):
        m_ayla *= beta1
        m_ayla += (1-beta1)*grad_ayla
        v_ayla *= beta2
        v_ayla += (1-beta2)*(grad_ayla**2)
        m_hat_ayla = m_ayla / (1 - beta1**epoch)
        v_hat_ayla = v_ayla / (1 - beta2**epoch)
        param_ayla -= lr * m_hat_ayla / (np.sqrt(v_hat_ayla) + eps)

# ---------- Plot Loss ----------
plt.figure(figsize=(10,6))
plt.plot(adam_losses, label='Standard ADAM', color='red')
plt.plot(ayla_losses, label='AYLA-ADAM', color='blue')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Comparison of ADAM vs AYLA")
plt.grid()
plt.show()

# ---------- Plot Predictions ----------
plt.figure(figsize=(10,6))
plt.scatter(X_input, y_input, label='True Data', color='red', alpha=0.6)

X_grid = np.linspace(-1, 3, 200).reshape(-1,1)
a1_grid_adam = np.tanh(X_grid @ W1_adam + b1_adam)
y_pred_grid_adam = a1_grid_adam @ W2_adam + b2_adam

a1_grid_ayla = np.tanh(X_grid @ W1_ayla + b1_ayla)
y_pred_grid_ayla = a1_grid_ayla @ W2_ayla + b2_ayla

plt.plot(X_grid, y_pred_grid_adam, label='ADAM Prediction', color='red')
plt.plot(X_grid, y_pred_grid_ayla, label='AYLA Prediction', color='blue')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Prediction Comparison")
plt.grid()
plt.show()
