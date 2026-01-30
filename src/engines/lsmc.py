from instruments import Option
from typing import Protocol
import numpy as np

class Basis(Protocol): 
    def prepare_basis(self, S_t: np.ndarray) -> np.ndarray: pass

        
class Polynomial:
    def __init__(self, _degree: int) -> None:
        self.degree = _degree
    
    def prepare_basis(self, S_t: np.ndarray) -> np.ndarray:
        return np.column_stack([S_t ** i for i in range(self.degree + 1)])


class LSCMEngine:
    def __init__(self, _basis: Basis):
        self.basis = _basis
        self.betas = None
    
    def compute_betas(self, S: np.ndarray, option: Option) -> float:
        rate: float = option.getRate()
        maturities: np.ndarray = option.getMaturities()

        n_times: int
        n_paths: int
        n_times, n_paths = S.shape

        self.betas = [None] * n_times

        # Terminal payoff
        Y: np.ndarray = option.payoff(S[-1])

        # Backward induction
        for i in range(n_times - 2, 0, -1):
            dt: float = maturities[i + 1] - maturities[i]
            discount: float = np.exp(-rate * dt)

            Y = Y * discount

            payoff: np.ndarray = option.payoff(S[i])
            itm: np.ndarray = payoff > 0

            if np.any(itm):
                X: np.ndarray = self.basis.prepare_basis(S[i, itm])

                beta: np.ndarray = np.linalg.lstsq(X, Y[itm], rcond=None)[0]
                continuation_vals: np.ndarray = X @ beta

                should_exercise: np.ndarray = payoff[itm] >= continuation_vals

                # Update cashflows
                Y[itm] = np.where(should_exercise, payoff[itm], Y[itm])

                # Store betas for future use
                self.betas[i] = beta

        # Final discounting
        Y = Y * np.exp(-rate * maturities[1])

        return np.mean(Y)
    
    def price(self, S: np.ndarray, option: Option) -> float:
        if self.betas is None:
            raise ValueError("Regression coefficients not computed; call compute_betas method first.")
        
        rate: float = option.getRate()
        maturities: np.ndarray = option.getMaturities()

        n_times: int
        n_paths: int
        n_times, n_paths = S.shape

        # Terminal payoff
        Y: np.ndarray = option.payoff(S[-1])

        # Backward induction
        for i in range(n_times - 2, 0, -1):
            dt: float = maturities[i + 1] - maturities[i]
            discount: float = np.exp(-rate * dt)

            Y = Y * discount

            payoff: np.ndarray = option.payoff(S[i])
            itm: np.ndarray = payoff > 0

            if np.any(itm) and self.betas[i] is not None:
                X: np.ndarray = self.basis.prepare_basis(S[i, itm])
                continuation_vals: np.ndarray = X @ self.betas[i]

                should_exercise: np.ndarray = payoff[itm] >= continuation_vals

                # Update cashflows
                Y[itm] = np.where(should_exercise, payoff[itm], Y[itm])

        # Final discounting
        Y = Y * np.exp(-rate * maturities[1])

        return np.mean(Y)