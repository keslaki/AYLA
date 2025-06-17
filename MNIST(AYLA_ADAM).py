#MNIST  NN not CNN

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ------------------- Hyperparameters ------------------- #
lr = float(input("Enter learning rate (e.g. 0.001): "))
N1 = float(input("Enter N1 for AYLA (e.g. 1.4): "))
N2 = float(input("Enter N2 for AYLA (e.g. 1.0): "))
epochs = 20
batch_size = 256

# ------------------- Load MNIST ------------------- #
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0  # Flatten to vectors and normalize
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# ------------------- NN Model ------------------- #
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(28*28,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


model_adam = build_model()
model_ayla = tf.keras.models.clone_model(model_adam)
model_ayla.set_weights(model_adam.get_weights())

# Optimizers
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=lr)
optimizer_ayla = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# ------------------- Training Loop ------------------- #
history = {
    'adam_train_loss': [], 'adam_train_acc': [],
    'adam_test_loss': [], 'adam_test_acc': [],
    'ayla_train_loss': [], 'ayla_train_acc': [],
    'ayla_test_loss': [], 'ayla_test_acc': []
}

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)

for epoch in range(epochs):
    # Adam training
    for xb, yb in train_ds:
        with tf.GradientTape() as tape:
            y_pred = model_adam(xb, training=True)
            loss = loss_fn(yb, y_pred)
        grads = tape.gradient(loss, model_adam.trainable_variables)
        optimizer_adam.apply_gradients(zip(grads, model_adam.trainable_variables))
        
    # AYLA training
    for xb, yb in train_ds:
        with tf.GradientTape() as tape:
            y_pred = model_ayla(xb, training=True)
            loss = loss_fn(yb, y_pred)
        grads = tape.gradient(loss, model_ayla.trainable_variables)
        # AYLA scaling
        loss_np = loss.numpy()
        condition = np.abs(loss_np) > 1
        condition2 = np.abs(loss_np) < 1
        nnp = np.where(condition, N2, np.where(condition2, N1, 1))
        factor = nnp * (np.abs(loss_np))**(nnp - 1)
        grads_scaled = [g * factor for g in grads]
        optimizer_ayla.apply_gradients(zip(grads_scaled, model_ayla.trainable_variables))
    
    # Evaluation
    def evaluate(model, x_data, y_data):
        y_pred = model(x_data, training=False)
        loss = loss_fn(y_data, y_pred).numpy()
        acc = np.mean(np.argmax(y_pred.numpy(), axis=1) == np.argmax(y_data, axis=1))
        return loss, acc

    train_loss_adam, train_acc_adam = evaluate(model_adam, x_train, y_train)
    test_loss_adam, test_acc_adam = evaluate(model_adam, x_test, y_test)
    train_loss_ayla, train_acc_ayla = evaluate(model_ayla, x_train, y_train)
    test_loss_ayla, test_acc_ayla = evaluate(model_ayla, x_test, y_test)

    # Store
    history['adam_train_loss'].append(train_loss_adam)
    history['adam_train_acc'].append(train_acc_adam)
    history['adam_test_loss'].append(test_loss_adam)
    history['adam_test_acc'].append(test_acc_adam)
    history['ayla_train_loss'].append(train_loss_ayla)
    history['ayla_train_acc'].append(train_acc_ayla)
    history['ayla_test_loss'].append(test_loss_ayla)
    history['ayla_test_acc'].append(test_acc_ayla)

    print(f"Epoch {epoch+1}: Adam [TrainAcc {train_acc_adam:.4f}, TestAcc {test_acc_adam:.4f}] | "
          f"AYLA [TrainAcc {train_acc_ayla:.4f}, TestAcc {test_acc_ayla:.4f}]")


# ------------------- Plotting ------------------- #
plt.plot(history['adam_train_loss'], label='Adam Train', color='black', linestyle='-')
plt.plot(history['adam_test_loss'], label='Adam Test', color='black', linestyle='--')
plt.plot(history['ayla_train_loss'], label='AYLA Train', color='blue', linestyle='-')
plt.plot(history['ayla_test_loss'], label='AYLA Test', color='blue', linestyle='--')
plt.title("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss.png", dpi=300)
plt.show()

plt.plot(history['adam_train_acc'], label='Adam Train', color='black', linestyle='-')
plt.plot(history['adam_test_acc'], label='Adam Test', color='black', linestyle='--')
plt.plot(history['ayla_train_acc'], label='AYLA Train', color='blue', linestyle='-')
plt.plot(history['ayla_test_acc'], label='AYLA Test', color='blue', linestyle='--')
plt.title("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Accu.png", dpi=300)
plt.show()
