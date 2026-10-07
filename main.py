# Main entry point for the Neural Network Decision Boundary Visualizer.
from data import generate_concentric_circles
from model import NeuralNetwork
from visualizer import run_visualizer

def main():
    print("Initializing Neural Network Decision Boundary Visualizer")
    
    # Hyperparameters and Configuration
    n_samples = 600
    noise = 0.08
    learning_rate = 0.1
    epochs = 1500
    
    # Architecture: [Input features, Hidden Layer 1, Hidden Layer 2, Output]
    layer_sizes = [2, 16, 16, 1]
    
    # Generate Dataset
    print(f"Generating dataset ({n_samples} samples)")
    X, y = generate_concentric_circles(n_samples=n_samples, noise=noise)
    
    # Initialize Model
    print(f"Building MLP with architecture: {layer_sizes}")
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate)
    
    # Run Training & Real-Time Visualization
    print("Starting training loop and visualizer")
    run_visualizer(nn, X, y, epochs=epochs)

if __name__ == "__main__":
    main()