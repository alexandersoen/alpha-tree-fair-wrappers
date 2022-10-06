import dill
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, List
from dataclasses import dataclass

# Generic types for the input, intended to be a dataclass
S = TypeVar('S')
T = TypeVar('T')

# Data class for protected information
@dataclass
class ProtectedInfo:
    protected_columns: List[int]
    unprotected_columns: List[int]
    protected_domain: List[np.ndarray]

    # Extra information for optimization
    enumerate_columns: List[int]

# Classifier types and abstract class
class Classifier(ABC, Generic[S]):
    """ Abstract type for classifier. Used as a wrapper around any type of
        generic classifier.
    """

    @abstractmethod
    def predict(self, x: S) -> np.ndarray:
        pass

    def __call__(self, x: S) -> np.ndarray:
        return self.predict(x)

class ClippedClassifier(Classifier[S], ABC):
    B: float
    min_val: float
    max_val: float

    def __init__(self, B: float) -> None:
        super().__init__()

        self.B = B
        self.min_val = 1 / (1 + np.exp(self.B))
        self.max_val = 1 / (1 + np.exp(-self.B))

        print('Clipped min: {}; Clipped max: {}'.format(self.min_val, self.max_val))

    def predict(self, x: S):
        return np.clip(self._predict(x), a_min=self.min_val, a_max=self.max_val)

    @abstractmethod
    def _predict(self, x: S):
        pass

# Twister types and abstract class
class Twister(ABC, Generic[T]):
    """ Abstract type for twister g. Used as a wrapper around any tilter
        function.
    """

    @abstractmethod
    def twist(self, x: T) -> np.ndarray:
        pass

    def __call__(self, x: T) -> np.ndarray:
        return self.twist(x)

# Wrapper types and abstract class
class Wrapper(ABC, Generic[S, T]):
    blackbox: Classifier[S]
    twister: Twister[T]

    def __init__(self, blackbox: Classifier[S], twister: Twister[T]) -> None:
        self.blackbox = blackbox
        self.twister = twister
        super().__init__()

    def predict(self, xb: S, xt: T) -> np.ndarray:
        unfair_pred = self.blackbox.predict(xb)
        correction = self.twister.twist(xt)

        corrected = np.empty_like(correction)
        finite_correction = np.where(np.isfinite(correction), correction, 0)
        corrected = 1 / ( 1 + np.exp(finite_correction * (np.log(1 - unfair_pred) - np.log(unfair_pred))))
        corrected[correction == float('inf')] = 0
        corrected[correction == float('-inf')] = 1

        return corrected

    def __call__(self, xb: S, xt: T) -> np.ndarray:
        return self.predict(xb, xt)

    def save(self, file_path: Path):
        with open(file_path, 'wb') as f:
            dill.dump(self, f)