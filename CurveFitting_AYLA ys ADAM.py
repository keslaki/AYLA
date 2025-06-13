import numpy as np
import matplotlib.pyplot as plt

# ------------------ Hyperparameters ------------------- #
N1, N2 = 1.8, 0.8
numerbss, randomm = 100, 0.2
input_size, hidden_size, output_size = 1, 128, 1
learning_rate, epochs = 0.005, 400
beta1_base, beta2_base = 0.9, 0.999
epsilon = 1e-8  # safer for division stability

# ------------------ Data Generation ------------------- #
np.random.seed(42)
X = np.linspace(-1, 3, numerbss).reshape(-1, 1)
true_func = lambda x: (1/3)*x**4 - (4/3)*x**3 + x**2 + (2/3)*x - (2/3)
y = true_func(X) + np.random.normal(0, randomm, (numerbss, 1))


# ------------------ Model Initialization ------------------- #
def initialize_model():
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    return (W1, b1, W2, b2), (mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2)


# ------------------ Forward Propagation ------------------- #
def forward(X, W1, b1, W2, b2):
    h = np.dot(X, W1) + b1
    h_relu = np.maximum(0, h)
    y_pred = np.dot(h_relu, W2) + b2
    return h, h_relu, y_pred


# ------------------ AYLA Gradient Modification ------------------- #
def ayla_coeff(y_pred, y_true, N1, N2):
    diff = y_pred - y_true
    condition = np.abs(diff) > 1
    nnp = np.where(condition, N2, N1)
    coeff = nnp * (np.abs(diff))**(nnp - 1)
    return coeff, diff, nnp


# ------------------ Training Function ------------------- #
def train(X, y, optimizer='adam'):
    (W1, b1, W2, b2), (mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2) = initialize_model()
    losses = []

    for t in range(1, epochs+1):
        h, h_relu, y_pred = forward(X, W1, b1, W2, b2)

        if optimizer == 'ayla':
            coeff, diff, nnp = ayla_coeff(y_pred, y, N1, N2)
            loss = np.mean((y_pred - y)**2)
            dy_pred = 2 * coeff * (y_pred - y) / X.shape[0]

            # dynamic beta update (AYLA-specific)
            beta1 = 1 - (1 - beta1_base) * np.mean(coeff)
            beta2 = 1 - (1 - beta2_base) * (np.mean(coeff))**2
            B1 = 1 - (1 - beta1_base**t) * np.mean(coeff)
            B2 = 1 - (1 - beta2_base**t) * (np.mean(coeff))**2          
            
        else:  # Adam
            loss = np.mean((y_pred - y)**2)
            dy_pred = 2 * (y_pred - y) / X.shape[0]
            beta1, beta2 = beta1_base, beta2_base
            B1, B2 = beta1**t, beta2**t

        # Gradients
        dW2 = np.dot(h_relu.T, dy_pred)
        db2 = np.sum(dy_pred, axis=0)
        d_hidden = np.dot(dy_pred, W2.T) * (h > 0)
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)

        # Adam update
        for param, grad, m, v, name in [
            (W1, dW1, mW1, vW1, 'W1'), (b1, db1, mb1, vb1, 'b1'),
            (W2, dW2, mW2, vW2, 'W2'), (b2, db2, mb2, vb2, 'b2')
        ]:
            m[:] = beta1 * m + (1 - beta1_base) * grad
            v[:] = beta2 * v + (1 - beta2_base) * grad**2
            m_hat = m / (1 - B1)
            v_hat = v / (1 - B2)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        losses.append(loss)

        if t % 100 == 0:
            print(f"{optimizer.upper()} Epoch {t} - Loss: {loss:.5f}")

    return (W1, b1, W2, b2), losses


# ------------------ Train both models ------------------- #
model_adam, losses_adam = train(X, y, optimizer='adam')
model_ayla, losses_ayla = train(X, y, optimizer='ayla')

# ------------------ Prediction ------------------- #
_, _, y_pred_adam = forward(X, *model_adam)
_, _, y_pred_ayla = forward(X, *model_ayla)

# ------------------ Final Plotting ------------------- #
plt.figure(figsize=(14, 6))

# Plot 1: Curve Fitting
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='black', alpha=0.5, label='Data')
plt.plot(X, y_pred_adam, 'r-', label='Adam')
plt.plot(X, y_pred_ayla, 'b-', label='AYLA')
plt.plot(X, true_func(X), 'k--', label='True function')
plt.title("Curve Fitting")
plt.legend()
plt.grid(True, linestyle='dotted')

# Plot 2: Loss Curves
plt.subplot(1, 2, 2)
plt.plot(losses_adam, 'r-', label='Adam Loss')
plt.plot(losses_ayla, 'b-', label='AYLA Loss')
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True, linestyle='dotted')

plt.tight_layout()
plt.savefig("AYLA_vs_ADAM.png", dpi=300)
plt.show()

print(f"Final Adam Loss: {losses_adam[-1]:.5f}")
print(f"Final AYLA Loss: {losses_ayla[-1]:.5f}")
