import numpy as np

# Activation Functions and Derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    # Clip to prevent overflow issues
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Initializing weights and biases using He/Xavier-like random scaling
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            # Weight matrix shape: (inputs_to_layer, neurons_in_layer)
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)


    # Performs forward pass
    def forward(self, X):
        
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        # Hidden layers use ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_input = relu(z)
            self.activations.append(current_input)
            
        # Output layer uses Sigmoid for binary classification
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = sigmoid(z_out)
        self.activations.append(output)
        
        return output

    # Binary cross-entropy loss
    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss


    # Performs backpropagation
    def backward(self, y_true):

        m = y_true.shape[0]  # Number of samples
        L = len(self.weights) - 1  # Index of output layer
        
        # Gradient of Binary Cross-Entropy with respect to Sigmoid output
        # dL/d(y_pred) = (y_pred - y_true) / (y_pred * (1 - y_pred)) combined with sigmoid derivative
        dz = self.activations[-1] - y_true
        
        for i in range(L, -1, -1):
            # Compute gradients for weights and biases of layer i
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # If not the first layer, compute dz for the previous layer using weights
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * relu_derivative(self.z_values[i-1])
                
            # Gradient descent update step
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    # Executes a single forward pass, loss calculation, and backward update step.
    def train_step(self, X, y):
        
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        self.backward(y)
        return loss