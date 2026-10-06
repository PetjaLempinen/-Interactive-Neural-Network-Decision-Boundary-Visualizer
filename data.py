import numpy as np

def generate_concentric_circles(n_samples=500, noise=0.05, factor=0.5):
    
    np.random.seed(42)
    
    # Half samples for inner circle and half for outer ring
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner
    
    # Inner circle
    r_inner = np.random.uniform(0.0, 0.3, n_inner)
    theta_inner = np.random.uniform(0.0, 2 * np.pi, n_inner)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    y_inner = np.zeros((n_inner, 1))
    
    # Outer ring
    r_outer = np.random.uniform(0.6, 1.0, n_outer)
    theta_outer = np.random.uniform(0.0, 2 * np.pi, n_outer)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    y_outer = np.ones((n_outer, 1))
    
    # Combine data and add slight noise
    X = np.vstack([X_inner, X_outer])
    y = np.vstack([y_inner, y_outer])
    
    X += np.random.normal(0, noise, X.shape)
    
    return X, y

if __name__ == "__main__":
    # test
    X, y = generate_concentric_circles()
    print("Data shapes - X:", X.shape, "y:", y.shape)