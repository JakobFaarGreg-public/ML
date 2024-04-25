from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    """Base class for a machine learning classifier"""

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        return False

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return False

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        return False
