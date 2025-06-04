import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time  # Added to measure training time

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Get user inputs at the beginning
EPOCHS = int(input("Enter number of epochs: "))
N1 = float(input("Enter N1 value: "))
N2 = float(input("Enter N2 value: "))

epsilon = 1e-7
BATCH_SIZE = 128

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Model definition
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD()

# Save initial weights
base_model = create_model()
initial_weights = base_model.get_weights()

model_modified = create_model()
model_modified.set_weights(initial_weights)

model_normal = create_model()
model_normal.set_weights(initial_weights)

# Helper functions
def sign(x):
    return tf.sign(x)

def compute_gradient_norm(grads):
    norm = 0.0
    for g in grads:
        if g is not None:
            norm += tf.reduce_sum(tf.square(g))
    return tf.sqrt(norm).numpy()

# Custom training loop with modifications
def train_with_custom_modification(epochs=EPOCHS, batch_size=BATCH_SIZE, model=None, N1=N1, N2=N2):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'grad_norm': []}
    
    for epoch in range(epochs):
        epoch_loss = []
        epoch_grad_norm = []
        epoch_acc = []
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, preds)
                loss_scalar = tf.reduce_mean(loss_value)
                loss_sign = sign(loss_scalar)
                abs_loss = tf.abs(loss_scalar)
                
                # Apply loss modification
                if abs_loss > 1:
                    loss_mod = loss_sign * ((abs_loss + epsilon) ** N1)
                    grad_scale = N1
                    grad_exponent = N1 - 1
                elif abs_loss < 1:
                    loss_mod = loss_sign * ((abs_loss + epsilon) ** N2)
                    grad_scale = N2
                    grad_exponent = N2 - 1
                else:
                    loss_mod = loss_scalar
                    grad_scale = 1.0
                    grad_exponent = 0.0

                loss_to_optimize = loss_mod

            grads = tape.gradient(loss_to_optimize, model.trainable_variables)
            grad_norm = compute_gradient_norm(grads)
            epoch_grad_norm.append(grad_norm)

            # Modify gradients
            new_grads = []
            for g in grads:
                if g is None:
                    new_grads.append(g)
                    continue
                g_sign = sign(g)
                abs_g = tf.abs(g)
                g_new = grad_scale * g_sign * tf.where(
                    tf.math.is_finite(abs_g),
                    abs_g * ((abs_g + epsilon) ** grad_exponent),
                    tf.zeros_like(abs_g)
                )
                new_grads.append(g_new)

            if all(g is not None for g in new_grads):
                optimizer.apply_gradients(zip(new_grads, model.trainable_variables))

            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_batch, axis=1), tf.argmax(preds, axis=1)), tf.float32))
            epoch_loss.append(loss_scalar.numpy())
            epoch_acc.append(acc.numpy())

        # Validation
        val_preds = model(x_test, training=False)
        val_loss = loss_fn(y_test_cat, val_preds).numpy()
        val_acc = np.mean(np.argmax(val_preds.numpy(), axis=1) == y_test)
        print(f"AYLA    - Epoch {epoch+1}: Loss={np.mean(epoch_loss):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Grad Norm={np.mean(epoch_grad_norm):.4f}")

        history['loss'].append(np.mean(epoch_loss))
        history['val_loss'].append(val_loss)
        history['accuracy'].append(np.mean(epoch_acc))
        history['val_accuracy'].append(val_acc)
        history['grad_norm'].append(np.mean(epoch_grad_norm))

    return history

# Standard training loop
def train_normal(epochs=EPOCHS, batch_size=BATCH_SIZE, model=None):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'grad_norm': []}

    for epoch in range(epochs):
        epoch_loss = []
        epoch_grad_norm = []
        epoch_acc = []

        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, preds)

            grads = tape.gradient(loss_value, model.trainable_variables)
            grad_norm = compute_gradient_norm(grads)
            epoch_grad_norm.append(grad_norm)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(tf.reduce_mean(loss_value).numpy())
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_batch, axis=1), tf.argmax(preds, axis=1)), tf.float32))
            epoch_acc.append(acc.numpy())

        val_preds = model(x_test, training=False)
        val_loss = loss_fn(y_test_cat, val_preds).numpy()
        val_acc = np.mean(np.argmax(val_preds.numpy(), axis=1) == y_test)
        print(f"Normal - Epoch {epoch+1}: Loss={np.mean(epoch_loss):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Grad Norm={np.mean(epoch_grad_norm):.4f}")

        history['loss'].append(np.mean(epoch_loss))
        history['val_loss'].append(val_loss)
        history['accuracy'].append(np.mean(epoch_acc))
        history['val_accuracy'].append(val_acc)
        history['grad_norm'].append(np.mean(epoch_grad_norm))

    return history

# === Run and Time Training ===
start_time_mod = time.time()
history_mod = train_with_custom_modification(epochs=EPOCHS, model=model_modified, N1=N1, N2=N2)
end_time_mod = time.time()
print(f"\nCustom Modified Model Training Time: {end_time_mod - start_time_mod:.2f} seconds")

start_time_norm = time.time()
history_norm = train_normal(epochs=EPOCHS, model=model_normal)
end_time_norm = time.time()
print(f"\nNormal Model Training Time: {end_time_norm - start_time_norm:.2f} seconds")


## PLOT

# Plot comparison
plt.figure(figsize=(6, 4))

# Accuracy plot
#plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history_norm['accuracy'], label='Normal Train', color='black', linestyle='-')
plt.plot(range(1, EPOCHS + 1), history_norm['val_accuracy'], label='Normal Val', color='black', linestyle='--')
plt.plot(range(1, EPOCHS + 1), history_mod['accuracy'], label='AYLA Train', color='blue', linestyle='-')
plt.plot(range(1, EPOCHS + 1), history_mod['val_accuracy'], label='AYLA Val', color='blue', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy (N1={N1}, N2={N2})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('SM1_acc.png', dpi=300)
plt.show()


plt.figure(figsize=(6, 4))
# Loss plot
#plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history_norm['loss'], label='Normal Loss', color='black', linestyle='-')
plt.plot(range(1, EPOCHS + 1), history_norm['val_loss'], label='Normal Val Loss', color='black', linestyle='--')
plt.plot(range(1, EPOCHS + 1), history_mod['loss'], label='AYLA Loss', color='blue', linestyle='-')
plt.plot(range(1, EPOCHS + 1), history_mod['val_loss'], label='AYLA Val Loss', color='blue', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss (N1={N1}, N2={N2})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('SM1_loss.png', dpi=300)
plt.show()


