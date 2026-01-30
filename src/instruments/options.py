from abc import ABC, abstractmethod
import numpy as np 

class Option(ABC):
    def __init__(self, _rate: float, _strike: float, _maturities: np.ndarray) -> None:
       self.rate = _rate
       self.strike = _strike
       self.maturities = _maturities

    @abstractmethod
    def payoff(self, S_i: np.ndarray) -> float:
        pass

    def getRate(self) -> float:
        return self.rate
    
    def getMaturities(self) -> np.ndarray:
        return self.maturities

class BermudanCall(Option):
    def __init__(self, _rate: float, _strike: float, _maturities: np.ndarray) -> None:
        super().__init__(_rate, _strike, _maturities)

    def payoff(self, S_i: np.ndarray) -> float:
        return np.maximum(S_i - self.strike, 0.0)

class BermudanPut(Option):
    def __init__(self, _rate: float, _strike: float, _maturities: np.ndarray) -> None:
        super().__init__(_rate, _strike, _maturities)
    
    def payoff(self, S_i: np.ndarray) -> float:
        return np.maximum(self.strike - S_i, 0.0)