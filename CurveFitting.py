import numpy as np
import matplotlib.pyplot as plt

# ---------- Generate Data ----------
np.random.seed(42)
num_points, noise = 100, 0.2
X_input = np.linspace(-1, 3, num_points).reshape(-1, 1)
y_true = (1/3)*X_input**4 - (4/3)*X_input**3 + X_input**2 + (2/3)*X_input - (2/3)
y_true += np.random.normal(0, noise, (num_points, 1))

# ---------- Hyperparameters ----------
beta1, beta2 = 0.9, 0.999
eps = 1e-8
lr = 0.01
epochs = 200
hidden_dim = 128

# ---------- Model Parameters ----------
def init_params():
    W1 = np.random.randn(1, hidden_dim) * 0.5
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * 0.5
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# ---------- AYLA Settings ----------
ayla_settings = [
    ((0.1, 0.1), 'blue', '-.', "AYLA (N1=0.1, N2=0.1)"),
    ((0.5, 0.5), 'green', '--', "AYLA (N1=0.5, N2=0.5)"),
    ((0.5, 1.5), 'orange', '-.', "AYLA (N1=0.5, N2=1.5)"),
    ((1.5, 0.5), 'purple', ':', "AYLA (N1=1.5, N2=0.5)")
]

loss_results = {}
predictions = {}

# ---------- Train ADAM ----------
W1, b1, W2, b2 = init_params()
mW1 = np.zeros_like(W1)
mW2 = np.zeros_like(W2)
mb1 = np.zeros_like(b1)
mb2 = np.zeros_like(b2)
vW1 = np.zeros_like(W1)
vW2 = np.zeros_like(W2)
vb1 = np.zeros_like(b1)
vb2 = np.zeros_like(b2)
adam_losses = []

for epoch in range(1, epochs+1):
    z1 = X_input @ W1 + b1
    a1 = np.tanh(z1)
    y_pred = a1 @ W2 + b2
    loss = y_pred - y_true
    mse = np.mean(loss**2)
    adam_losses.append(mse)

    grad = 2 * loss / num_points
    dW2 = a1.T @ grad
    db2 = np.sum(grad, axis=0, keepdims=True)
    da1 = grad @ W2.T
    dz1 = da1 * (1 - a1**2)
    dW1 = X_input.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    for g, m, v, p in zip([dW1, db1, dW2, db2], [mW1, mb1, mW2, mb2], [vW1, vb1, vW2, vb2], [W1, b1, W2, b2]):
        m *= beta1
        m += (1 - beta1) * g
        v *= beta2
        v += (1 - beta2) * g**2
        m_hat = m / (1 - beta1**epoch)
        v_hat = v / (1 - beta2**epoch)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

loss_results['ADAM'] = adam_losses
a1_grid = np.tanh(np.linspace(-1, 3, 200).reshape(-1,1) @ W1 + b1)
predictions['ADAM'] = a1_grid @ W2 + b2

# ---------- Train AYLA Variants ----------
for (N1, N2), color, style, label in ayla_settings:
    W1, b1, W2, b2 = init_params()
    mW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2)
    mb1 = np.zeros_like(b1)
    mb2 = np.zeros_like(b2)
    vW1 = np.zeros_like(W1)
    vW2 = np.zeros_like(W2)
    vb1 = np.zeros_like(b1)
    vb2 = np.zeros_like(b2)
    ayla_losses = []

    for epoch in range(1, epochs+1):
        z1 = X_input @ W1 + b1
        a1 = np.tanh(z1)
        y_pred = a1 @ W2 + b2
        loss = y_pred - y_true
        mse = np.mean(loss**2)
        ayla_losses.append(mse)

        abs_loss = np.abs(mse)
        nnp = N2 if abs_loss > 1 else N1
        factor = nnp * abs_loss**(nnp - 1)
        grad = 2 * factor * loss / num_points

        dW2 = a1.T @ grad
        db2 = np.sum(grad, axis=0, keepdims=True)
        da1 = grad @ W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X_input.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        for g, m, v, p in zip([dW1, db1, dW2, db2], [mW1, mb1, mW2, mb2], [vW1, vb1, vW2, vb2], [W1, b1, W2, b2]):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g**2
            m_hat = m / (1 - beta1**epoch)
            v_hat = v / (1 - beta2**epoch)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    loss_results[label] = ayla_losses
    a1_grid = np.tanh(np.linspace(-1, 3, 200).reshape(-1,1) @ W1 + b1)
    predictions[label] = a1_grid @ W2 + b2

# ---------- Plot Losses ----------
plt.figure(figsize=(8, 5))
#plt.xscale("log")
plt.plot(loss_results['ADAM'], label='ADAM', color='red', linestyle='-')
for (N1, N2), color, style, label in ayla_settings:
    plt.plot(loss_results[label], label=label, color=color, linestyle=style)
plt.xlabel("Epochs (log scale)")
plt.ylabel("MSE Loss")
plt.title("Loss vs Epochs (log x-axis)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LogLoss.png", dpi=300)
plt.show()

# ---------- Plot Predictions ----------
plt.figure(figsize=(8, 5))
plt.scatter(X_input, y_true, label='True Data', color='grey', alpha=0.5)
X_grid = np.linspace(-1, 3, 200).reshape(-1,1)
plt.plot(X_grid, predictions['ADAM'], label='ADAM', color='red', linestyle='-')
for (_, _, _, label), color, style in zip(ayla_settings, [s[1] for s in ayla_settings], [s[2] for s in ayla_settings]):
    plt.plot(X_grid, predictions[label], label=label, color=color, linestyle=style)
plt.xlabel("X")
plt.ylabel("Predicted y")
plt.title("Prediction Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PredictionComparison.png", dpi=300)
plt.show()

# ---------- Final Losses ----------
print("\nFinal MSE Losses:")
for label, losses in loss_results.items():
    print(f"{label:25}: {losses[-1]:.6f}")


# ---------- Plot First N Epochs (Custom) ----------

first_elements = 20  # Set how many elements to show here

plt.figure(figsize=(8, 5))

# ADAM
plt.plot(range(1, first_elements + 1), loss_results['ADAM'][:first_elements],
         label='ADAM', color='red', linestyle='-')

# AYLA Variants
for (_, _, _, label), color, style in zip(ayla_settings, [s[1] for s in ayla_settings], [s[2] for s in ayla_settings]):
    plt.plot(range(1, first_elements + 1), loss_results[label][:first_elements],
             label=label, color=color, linestyle=style)

plt.xlabel(f"Epochs (First {first_elements})")
plt.ylabel("MSE Loss")
plt.title(f"Loss (First {first_elements} Epochs)")
plt.xticks(range(1, first_elements + 1, max(1, first_elements // 10)))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"First{first_elements}Epochs.png", dpi=300)
plt.show()
