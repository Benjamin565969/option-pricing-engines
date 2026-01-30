import numpy as np


def simulate_GBM(
    num_paths: int,
    time_grid: np.ndarray,
    S0: float,
    mu: float,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:
    """
    """
    dt = np.diff(time_grid)
    rng = np.random.default_rng(seed)
    
    # Simulate standard normal
    dW = rng.normal(0, 1, size=(num_paths, len(dt)))
    
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW
    
    S = np.empty((num_paths, len(time_grid)))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    return S