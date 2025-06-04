# MNIST Classification with Custom Loss Modification

## Overview

This project implements a neural network-based classification pipeline for the MNIST dataset, comparing a standard training approach with a custom-modified training loop. The modified loop applies loss scaling based on user-defined parameters `N1` and `N2`, and adjusts gradients accordingly. The script trains two models (Normal and AYLA), measures training time, and visualizes accuracy and loss metrics with customizable line colors and styles.

## Features

- **Custom Loss Modification**: Applies loss scaling based on the absolute loss value, using exponents `N1` (for losses > 1) and `N2` (for losses < 1).
- **Gradient Scaling**: Modifies gradients during training to match the loss scaling, enhancing model optimization.
- **User Input**: Prompts users for the number of epochs, `N1`, and `N2` values at runtime.
- **Training Time Measurement**: Records and reports training duration for both Normal and AYLA models.
- **Visualization**: Generates plots for accuracy and loss, comparing Normal and AYLA models with customizable line colors and styles.
- **Reproducibility**: Sets random seeds for NumPy and TensorFlow to ensure consistent results.

## Requirements

- Python 3.8 or higher
- Required Python packages:
  ```bash
  pip install tensorflow numpy matplotlib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/keslaki/AYLA.git
   cd AYLA
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install packages manually:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Ensure the `AYLA_MNIST.py` script is in the project directory.

## Usage

1. **Run the Script**:
   Execute the script to train and evaluate models on the MNIST dataset:
   ```bash
   python AYLA_MNIST.py
   ```

2. **Provide Inputs**:
   - When prompted, enter:
     - Number of epochs (e.g., `50`).
     - `N1` value (e.g., `1.5`): Exponent for loss scaling when absolute loss > 1.
     - `N2` value (e.g., `0.5`): Exponent for loss scaling when absolute loss < 1.

3. **Configuration**:
   - Modify hyperparameters in `AYLA_MNIST.py` if needed:
     ```python
     BATCH_SIZE = 128  # Batch size for training
     epsilon = 1e-7    # Small constant to avoid numerical issues
     ```
   - Adjust model architecture in the `create_model` function:
     ```python
     def create_model():
         return tf.keras.Sequential([
             tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
             tf.keras.layers.Dense(128, activation='relu'),
             tf.keras.layers.Dense(256, activation='relu'),
             tf.keras.layers.Dense(10, activation='softmax')
         ])
     ```
   - Customize plot colors and line styles in the plotting section:
     ```python
     # Example: Change colors
     plt.plot(..., color='black', linestyle='-')  # Normal Train
     plt.plot(..., color='black', linestyle='--') # Normal Val
     plt.plot(..., color='blue', linestyle='-')   # AYLA Train
     plt.plot(..., color='blue', linestyle='--')  # AYLA Val
     ```

4. **Output**:
   - The script prints epoch-wise metrics (loss, validation loss, validation accuracy, gradient norm) for both models.
   - Reports total training time for Normal and AYLA models.
   - Saves two plots:
     - `SM1_acc.png`: Accuracy comparison.
     - `SM1_loss.png`: Loss comparison.
   - Example console output:
     ```
     Enter number of epochs: 50
     Enter N1 value: 1.5
     Enter N2 value: 0.5
    
