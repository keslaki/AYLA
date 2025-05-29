MNIST Neural Network with AYLA Modification
This repository contains a Python script (AYLA.py) for training and comparing two neural networks on the MNIST dataset: a standard (Normal) method and a custom method (AYLA) with modified loss and gradients. The script allows users to specify the number of epochs and modification parameters N1 and N2, and it generates plots to compare the training and validation performance of both methods.
Project Overview
The AYLA.py script trains two identical neural networks on the MNIST dataset:

Normal Method: Uses standard categorical cross-entropy loss and gradient descent (SGD).
AYLA Method: Applies a custom modification to the loss and gradients based on the loss value:
If |loss| > 1: loss = sign(loss) * (|loss| + ε)^N1, gradient = N1 * gradient * sign(gradient) * (|gradient| + ε)^(N1-1)
If |loss| < 1: loss = sign(loss) * (|loss| + ε)^N2, gradient = N2 * gradient * sign(gradient) * (|gradient| + ε)^(N2-1)
If |loss| = 1: No modification
A small ε = 0.0000001 is added to avoid numerical instability.



Both models start with the same initial weights for a fair comparison. The script outputs per-epoch metrics (loss, validation loss, validation accuracy, and gradient norm), displays an interactive plot comparing training and validation accuracy/loss, and saves the plot as mnist_comparison.png.
Features

User Inputs: Prompts for epochs, N1, and N2 at the start of the script.
Model Architecture: A neural network with a Flatten layer, a 128-unit Dense layer (ReLU activation), and a 10-unit Dense layer (softmax activation).
Training: Custom training loop for the AYLA method with modified loss and gradients; standard training for the Normal method.
Output:
Console output of per-epoch metrics (loss, validation loss, validation accuracy, gradient norm).
An interactive Matplotlib plot comparing training/validation accuracy and loss.
A saved plot (mnist_comparison.png).


Reproducibility: Fixed random seed (42) for consistent results.

Requirements

Python 3.6+
TensorFlow (tensorflow>=2.0)
NumPy (numpy)
Matplotlib (matplotlib)

Installation

Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install the required packages:
pip install tensorflow numpy matplotlib



Usage

Run the script:
python AYLA.py


Enter the prompted values:

Number of epochs: e.g., 5
N1 value: e.g., 0.95 (used when |loss| > 1)
N2 value: e.g., 1.0 (used when |loss| < 1)


The script will:

Train both models (Normal and AYLA) on the MNIST dataset.
Print per-epoch metrics:AYLA   - Epoch X: Loss=..., Val Loss=..., Val Acc=..., Grad Norm=...
Normal - Epoch X: Loss=..., Val Loss=..., Val Acc=..., Grad Norm=...


Display an interactive plot comparing training/validation accuracy and loss.
Save the plot as mnist_comparison.png in the working directory.



**File Structure
**
AYLA.py: Main Python script for training and plotting.
mnist_comparison.png: Output plot (generated after running the script).

**Notes**

Model Performance: The current architecture (128-unit hidden layer) is relatively simple. For improved accuracy on MNIST, consider increasing the hidden layer size or adding more layers in the create_model() function.
Numerical Stability: An ε = 0.0000001 is added to loss and gradient computations to prevent division by zero or numerical issues.
Gradient Norm: The script reports gradient norms for both methods, useful for debugging optimization stability.
Customization: Modify BATCH_SIZE (default: 128) or the model architecture in create_model() to experiment with different configurations.
Plot: The interactive plot is displayed using Matplotlib and saved as mnist_comparison.png. Ensure a graphical environment is available to view the plot.

Example
To train for 5 epochs with N1=0.95 and N2=1.0:
python AYLA.py

Enter:
Enter number of epochs: 5
Enter N1 value: 0.95
Enter N2 value: 1.0

The script will train the models, print metrics, display the plot, and save it as mnist_comparison.png.
Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.
**Contact**
For questions or issues, please open an issue on the GitHub repository or contact benkeslaki@gmail.com.
