import numpy as np
import matplotlib.pyplot as plt

# Set noise scale (if you want noisy gradients)
noise_scale = 0

# Define function and gradient
def f(x):
    return x**4 - 3 * x**3 + 2

def grad_f(x):
    return 4 * x**3 - 9 * x**2 + np.random.normal(0, noise_scale)

# SGD optimizer
def sgd(learning_rate=0.1, num_epochs=100, initial_x=1, N1=1.3, N2=1.9):
    x, x2 = initial_x, initial_x
    history = {
        'x_sgd': [], 'x_ayla': [],
        'y_sgd': [], 'y_ayla': [],
        'grad_sgd': [], 'grad_ayla': [],
        'step_sgd': [], 'step_ayla': []
    }

    for _ in range(num_epochs):
        v2 = abs(f(x2))
        nnp = 1 if v2 == 0 else (N1 if v2 < 1 else N2)

        # Record values
        history['x_sgd'].append(x)
        history['x_ayla'].append(x2)
        history['y_sgd'].append(f(x))
        history['y_ayla'].append(np.sign(f(x2)) * v2**nnp)

        # Compute gradients
        grad1 = grad_f(x)
        grad2 = grad_f(x2) * nnp * v2**(nnp-1)

        history['grad_sgd'].append(grad1)
        history['grad_ayla'].append(grad2)
        history['step_sgd'].append(learning_rate * grad1)
        history['step_ayla'].append(learning_rate * grad2)

        # Update steps
        x -= learning_rate * grad1
        x2 -= learning_rate * grad2

    return history

# Generate modified loss for full curve
def generate_loss_curve(f, N1, N2, x_range):
    y_sgd, y_ayla = [], []
    for x in x_range:
        v = abs(f(x))
        nnp = 1 if v == 0 else (N1 if v < 1 else N2)
        y_sgd.append(f(x))
        y_ayla.append(np.sign(f(x)) * v**nnp)
    return np.array(y_sgd), np.array(y_ayla)

# Plot optimization path
def plot_paths(history): 
        
    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.axhline(y=2.25, color='black', linestyle='-', linewidth=1, label='Absolute Minimum')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Saddle Point')

    plt.plot(history['x_sgd'], label='SGD', color='black', linewidth=2)
    plt.plot(history['x_ayla'], label='AYLA', color='blue', linewidth=2)
    plt.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('ayla_sgd_poly_1d.tif', dpi=1000, bbox_inches='tight')
    plt.show()

# Plot loss landscape and steps
def plot_loss_and_steps(history, N1, N2):
    xx = np.linspace(-1, 3, 600)
    yy_sgd, yy_ayla = generate_loss_curve(f, N1, N2, xx)

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.scatter(history['x_sgd'], history['y_sgd'], color='black', label='Steps(SGD)', s=22, facecolors='none', edgecolors='black', linewidth=1)
    ax.scatter(history['x_ayla'], history['y_ayla'], color='blue', label='Steps(AYLA)', s=22, facecolors='none', edgecolors='blue', linewidth=1, marker='s')
    plt.plot(xx, yy_sgd, color='black', label='Loss(SGD)', linewidth=1)
    plt.plot(xx, yy_ayla, color='blue', label='Loss(AYLA)', linewidth=1)

    plt.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('ayla_sgd_poly_2.tif', dpi=1000, bbox_inches='tight')
    plt.show()

# Hyperparameters
learning_rate = 0.03
num_epochs = 50
initial_x = -1
N1 = 1
N2 = 1.4

# Run optimization and plots
history = sgd(learning_rate, num_epochs, initial_x, N1, N2)
plot_paths(history)
plot_loss_and_steps(history, N1, N2)




# Plot learning_rate * gradients (step sizes)
def plot_step_sizes(history):
    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.plot(history['step_sgd'], label='SGD Step Size', color='black', linewidth=1.5)
    ax.plot(history['step_ayla'], label='AYLA Step Size', color='blue', linewidth=1.5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Step Size (learning_rate * gradient)', fontsize=10)
    plt.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='dotted', color='gray')
    plt.tight_layout()
    plt.savefig('ayla_sgd_step_sizes.tif', dpi=1000, bbox_inches='tight')
    plt.show()

plot_step_sizes(history)
