import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ------------------- Hyperparameters ------------------- #
lr = float(input("Enter learning rate (e.g. 0.001): "))
N1 = float(input("Enter N1 for AYLA (e.g. 1.4): "))
N2 = float(input("Enter N2 for AYLA (e.g. 1.0): "))
epochs = int(input("Enter EPOCH (e.g. 100): "))
batch_size = 128

# ------------------- Load MNIST ------------------- #
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# ------------------- Neural Network Model ------------------- #
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
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

# Training datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# History for plotting
history = {
    'adam_train_loss': [], 'adam_test_loss': [],
    'ayla_train_loss': [], 'ayla_test_loss': [],
    'adam_train_acc': [], 'adam_test_acc': [],
    'ayla_train_acc': [], 'ayla_test_acc': []
}

# ------------------- Training Loop ------------------- #
for epoch in range(epochs):
    # Training phase
    adam_train_loss, ayla_train_loss = 0.0, 0.0
    adam_train_acc, ayla_train_acc = 0.0, 0.0
    num_batches = 0

    for xb, yb in train_ds:
        num_batches += 1
        # Adam forward and backward pass
        with tf.GradientTape() as tape:
            y_pred_adam = model_adam(xb, training=True)
            loss_adam = loss_fn(yb, y_pred_adam)
        grads_adam = tape.gradient(loss_adam, model_adam.trainable_variables)
        optimizer_adam.apply_gradients(zip(grads_adam, model_adam.trainable_variables))

        # AYLA forward pass and scaling
        with tf.GradientTape() as tape:
            y_pred_ayla = model_ayla(xb, training=True)
            loss_ayla = loss_fn(yb, y_pred_ayla)
        grads_ayla = tape.gradient(loss_ayla, model_ayla.trainable_variables)

        # Use Adam loss for condition
        loss_adam_np = loss_adam.numpy()
        condition = tf.abs(loss_adam_np) > 1
        condition2 = tf.abs(loss_adam_np) < 1
        nnp = tf.where(condition, N2, tf.where(condition2, N1, 1))
        factor = nnp * tf.abs(loss_adam_np) ** (nnp - 1)

        # Scale gradients for AYLA
        grads_scaled = [g * factor for g in grads_ayla]
        optimizer_ayla.apply_gradients(zip(grads_scaled, model_ayla.trainable_variables))

        # Accumulate metrics
        adam_train_loss += loss_adam.numpy()
        ayla_train_loss += loss_ayla.numpy()
        adam_train_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred_adam, axis=1), tf.argmax(yb, axis=1)), tf.float32)).numpy()
        ayla_train_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred_ayla, axis=1), tf.argmax(yb, axis=1)), tf.float32)).numpy()

    # Average metrics over batches
    adam_train_loss /= num_batches
    ayla_train_loss /= num_batches
    adam_train_acc /= num_batches
    ayla_train_acc /= num_batches

    # Test phase
    adam_test_loss, ayla_test_loss = 0.0, 0.0
    adam_test_acc, ayla_test_acc = 0.0, 0.0
    num_test_batches = 0

    for xb, yb in test_ds:
        num_test_batches += 1
        y_pred_adam = model_adam(xb, training=False)
        y_pred_ayla = model_ayla(xb, training=False)
        adam_test_loss += loss_fn(yb, y_pred_adam).numpy()
        ayla_test_loss += loss_fn(yb, y_pred_ayla).numpy()
        adam_test_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred_adam, axis=1), tf.argmax(yb, axis=1)), tf.float32)).numpy()
        ayla_test_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred_ayla, axis=1), tf.argmax(yb, axis=1)), tf.float32)).numpy()

    adam_test_loss /= num_test_batches
    ayla_test_loss /= num_test_batches
    adam_test_acc /= num_test_batches
    ayla_test_acc /= num_test_batches

    # Store history
    history['adam_train_loss'].append(adam_train_loss)
    history['ayla_train_loss'].append(ayla_train_loss)
    history['adam_test_loss'].append(adam_test_loss)
    history['ayla_test_loss'].append(ayla_test_loss)
    history['adam_train_acc'].append(adam_train_acc)
    history['ayla_train_acc'].append(ayla_train_acc)
    history['adam_test_acc'].append(adam_test_acc)
    history['ayla_test_acc'].append(ayla_test_acc)

    # Compute gradient norms for printing (using last batch)
    with tf.GradientTape() as tape:
        y_pred_adam = model_adam(xb, training=True)
        loss_adam = loss_fn(yb, y_pred_adam)
    grads_adam = tape.gradient(loss_adam, model_adam.trainable_variables)
    grad_norm_adam = tf.reduce_mean([tf.norm(g) for g in grads_adam]).numpy()

    with tf.GradientTape() as tape:
        y_pred_ayla = model_ayla(xb, training=True)
        loss_ayla = loss_fn(yb, y_pred_ayla)
    grads_ayla = tape.gradient(loss_ayla, model_ayla.trainable_variables)
    grad_norm_ayla = tf.reduce_mean([tf.norm(g) for g in grads_ayla]).numpy()

    # Verbose output
    print(f"Epoch {epoch + 1}: "
          f"Adam [Train Loss: {adam_train_loss:.4f}, Test Loss: {adam_test_loss:.4f}, "
          f"Train Acc: {adam_train_acc:.4f}, Test Acc: {adam_test_acc:.4f}, Grad Norm: {grad_norm_adam:.4f}] | "
          f"AYLA [Train Loss: {ayla_train_loss:.4f}, Test Loss: {ayla_test_loss:.4f}, "
          f"Train Acc: {ayla_train_acc:.4f}, Test Acc: {ayla_test_acc:.4f}, Grad Norm: {grad_norm_ayla:.4f}, "
          f"Factor: {factor.numpy():.4f}]")

# ------------------- Plotting ------------------- #
plt.figure(figsize=(10, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history['adam_train_loss'], label='Adam Train Loss', color='black', linestyle='-')
plt.plot(history['adam_test_loss'], label='Adam Test Loss', color='black', linestyle='--')
plt.plot(history['ayla_train_loss'], label='AYLA Train Loss', color='blue', linestyle='-')
plt.plot(history['ayla_test_loss'], label='AYLA Test Loss', color='blue', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['adam_train_acc'], label='Adam Train Acc', color='black', linestyle='-')
plt.plot(history['adam_test_acc'], label='Adam Test Acc', color='black', linestyle='--')
plt.plot(history['ayla_train_acc'], label='AYLA Train Acc', color='blue', linestyle='-')
plt.plot(history['ayla_test_acc'], label='AYLA Test Acc', color='blue', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("mnist_comparison.png", dpi=300)
plt.show()
