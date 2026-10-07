import numpy as np
import matplotlib.pyplot as plt

def run_visualizer(nn, X, y, epochs=1500, update_every=15):
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Create a dense grid mesh for background decision boundary mapping creates 
    # 15000 invicible points for painting the background
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 150), np.linspace(-1.2, 1.2, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    for epoch in range(epochs):
        loss = nn.train_step(X, y)
        
        if epoch % update_every == 0 or epoch == epochs - 1:
            ax.clear()
            
            preds = nn.forward(grid_points)
            zz = preds.reshape(xx.shape)
            
            ax.contourf(xx, yy, zz, levels=25, cmap="coolwarm", alpha=0.7)
            ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5)
            
            ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], color="blue", edgecolor="k", label="Class 0", s=30)
            ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], color="red", edgecolor="k", label="Class 1", s=30)
            
            ax.set_title(f"Epoch {epoch} | Loss: {loss:.4f}", fontsize=12, fontweight="bold")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.legend(loc="upper right")
            
            plt.draw()
            plt.pause(0.001)
            
    plt.ioff()
    plt.show()