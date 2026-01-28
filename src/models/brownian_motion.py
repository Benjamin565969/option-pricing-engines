import numpy as np
import matplotlib.pyplot as plt


def simulate_BM(
    num_paths: int,
    time_grid: np.ndarray,
    drift: float = 0.0,
    seed: int = 42,
    X0: float = 0.0,
) -> np.ndarray:
    
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be a 1D array")
    
    if len(time_grid) < 2:
        raise ValueError("time_grid must contain at least 2 points")

    dt: np.ndarray = np.diff(time_grid)
    if np.any(dt <= 0):
        raise ValueError("time_grid must be strictly increasing")
    
    rng: np.Generator = np.random.default_rng(seed)

    # Simulate standard normal
    dZ: np.ndarray = rng.normal(loc=0, scale=1, size=(num_paths, len(dt)))

    # Add drift and diffusion
    dX: np.ndarray = drift * dt + np.sqrt(dt) * dZ

    # Build paths
    X: np.ndarray = np.empty(shape=(num_paths, len(time_grid)), dtype=float)
    X[:, 0] = X0
    X[:, 1:] = X0 + np.cumsum(dX, axis=1)
    
    return X