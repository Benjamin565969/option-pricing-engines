from engines.lsmc import LSCMEngine, Polynomial
from instruments.options import BermudanCall
from models.brownian_motion import simulate_GBM
import numpy as np

GRID: np.ndarray = np.linspace(start=0.0, stop=1.0, num=1000)
S0: float = 100.0
MU: float = 0.1
SIGMA: float = 0.25

sim1: np.ndarray = simulate_GBM(
    num_paths = 1000,
    time_grid = GRID,
    S0 = S0,
    mu = MU,
    sigma = SIGMA
)

sim2: np.ndarray = simulate_GBM(
    num_paths = 10000,
    time_grid = GRID,
    S0 = S0,
    mu = MU,
    sigma = SIGMA
)

option: BermudanCall = BermudanCall(_rate = 0.25, _strike = 10.0, _maturities = GRID)

polynomial_basis: Polynomial = Polynomial(_degree = 3)
longstaff_schwartz: LSCMEngine = LSCMEngine(_basis = polynomial_basis)

bias_price: float = longstaff_schwartz.compute_betas(sim1.T, option)
no_bias_price: float = longstaff_schwartz.price(sim2.T, option)

print(bias_price)
print(no_bias_price)