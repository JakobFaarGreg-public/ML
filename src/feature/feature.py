import pandas as pd
from abc import ABC, abstractmethod


class Feature(ABC):
    """Base class for a feature"""

    @abstractmethod
    def process(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def save(self):
        pass
