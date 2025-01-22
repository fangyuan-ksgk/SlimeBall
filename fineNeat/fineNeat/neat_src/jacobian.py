import numpy as np 

def estimate_jacobian(f, x, num_samples=100, epsilon=1e-4):
    """
    Estimate Jacobian matrix of vector-valued function f at point x using random directions.
    
    Args:
        f: Vector-valued function that returns numpy array
        x: Point at which to estimate Jacobian
        num_samples: Number of random directions to sample
        epsilon: Small perturbation for finite difference
    
    Returns:
        Estimated Jacobian matrix where J[i,j] = df_i/dx_j
    """
    x = np.asarray(x)
    f0 = np.asarray(f(x))
    n = len(x)      # Input dimension
    m = len(f0)     # Output dimension
    
    # Initialize Jacobian matrix
    J = np.zeros((m, n))
    
    # Coordinate-wise finite differences
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += epsilon
        
        x_minus = x.copy()
        x_minus[j] -= epsilon
        
        f_plus = np.asarray(f(x_plus))
        f_minus = np.asarray(f(x_minus))
        
        # Central difference
        J[:, j] = (f_plus - f_minus) / (2 * epsilon)
    
    return J

def estimate_jacobian_dg_naive(f, x, num_samples=100, epsilon=1e-4):
    d = len(x)  # input dimension
    v = np.random.randn(num_samples, len(x))
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)

    f_plus = np.array([f(x + v_normalized[i] * epsilon) for i in range(num_samples)])
    f_minus = np.array([f(x - v_normalized[i] * epsilon) for i in range(num_samples)])

    directional_gradient = (f_plus - f_minus) / (2 * epsilon)
    jacobian_estimate = d * np.mean(v_normalized[:, :, np.newaxis] * directional_gradient[:, np.newaxis, :], axis=0).T

    return jacobian_estimate

def generate_directions(num_samples, dimension):
    v = np.random.randn(num_samples, dimension)
    return v

def angle_to_rotation_vector(angles, dimension, max_freq=3):
    """
    Convert angles to rotation vector using Fourier basis
    
    Args:
        angles: (k,) array of base angles between 0 and 2π
        dimension: target dimension for parameter space
        max_freq: maximum frequency to use in Fourier series
    
    Returns:
        (k, dimension) array of rotation vectors
    """
    k = len(angles)
    v = np.zeros((k, dimension))
    
    # Generate Fourier components
    for i in range(dimension):
        freq = (i // 2) + 1  # Increasing frequencies
        if freq > max_freq:
            freq = ((i // 2) % max_freq) + 1  # Cycle through frequencies
            
        if i % 2 == 0:
            # Sine components
            v[:, i] = np.sin(freq * angles)
        else:
            # Cosine components
            v[:, i] = np.cos(freq * angles)
    
    # Normalize each vector
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    return v


def estimate_jacobian_dg(f, x, num_samples=100, epsilon=1e-4, angles=False):
    """
    Estimate Jacobian matrix using directional gradients with improved accuracy.
    """
    d = x.shape[-1]
    
    # Use minimum of num_samples and d
    k = min(num_samples, d)
    
    # Generate and normalize directions
    if angles:
        v = angle_to_rotation_vector(np.random.uniform(0, 2*np.pi, num_samples), d)
    else:
        v = generate_directions(num_samples, d)
        
    v = angle_to_rotation_vector(np.random.uniform(0, 2*np.pi, num_samples), d)
    q, r = np.linalg.qr(v.T)
    v_normalized = q.T[:k]  # Only take k directions
    
    f_plus = np.array([f(x + v_normalized[i] * epsilon) for i in range(k)])
    f_minus = np.array([f(x - v_normalized[i] * epsilon) for i in range(k)])
    
    directional_gradient = (f_plus - f_minus) / (2 * epsilon)
    
    # Use pseudoinverse when num_samples < d to handle underdetermined system
    jacobian_estimate = directional_gradient @ np.linalg.pinv(v_normalized[:k]).T
    # jacobian_estimate = directional_gradient.T @ np.linalg.pinv(v_normalized)

    return jacobian_estimate



def estimate_jacobian_entropy(f, x, num_samples=100, base_epsilon=1e-4, angles=False):
    """
    Estimate Jacobian matrix using directional gradients with entropy-based adaptive step sizes.
    """
    d = x.shape[-1]
    k = min(num_samples, d)
    
    # Generate initial directions as before
    if angles:
        v = angle_to_rotation_vector(np.random.uniform(0, 2*np.pi, num_samples), d)
    else:
        v = generate_directions(num_samples, d)
    q, r = np.linalg.qr(v.T)
    v_normalized = q.T
    
    # Sample points to estimate entropy
    f0 = f(x)
    test_samples = np.array([f(x + v_normalized[i] * base_epsilon) for i in range(k)])
    
    # Compute entropy-based scaling
    # Normalize samples to compute probabilities
    samples_flat = test_samples.reshape(-1, test_samples.shape[-1])
    samples_normalized = samples_flat - samples_flat.min(axis=0)
    probs = samples_normalized / (samples_normalized.sum(axis=0) + 1e-10)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Scale epsilon based on entropy: higher entropy → larger steps
    adaptive_epsilon = base_epsilon * (1 + entropy)
    
    # Use adaptive epsilon for Jacobian estimation
    f_plus = np.array([f(x + v_normalized[i] * adaptive_epsilon) for i in range(k)])
    f_minus = np.array([f(x - v_normalized[i] * adaptive_epsilon) for i in range(k)])
    
    directional_gradient = (f_plus - f_minus) / (2 * adaptive_epsilon)
    # Use pseudoinverse when num_samples < d
    jacobian_estimate = directional_gradient @ np.linalg.pinv(v_normalized[:k]).T
    g = jacobian_estimate / np.linalg.norm(jacobian_estimate)
    return g