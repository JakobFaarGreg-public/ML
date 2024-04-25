import numpy as np
import pandas as pd
from feature import Feature

class Sum(Feature):
    """A feature that sums all values"""

    def process(self, data: pd.DataFrame) -> np.ndarray:
        feature_vector = np.array([])
        for _, row in data.iterrows():
            feature_vector = np.append(feature_vector, [sum(row)])
        
        return feature_vector.reshape(len(feature_vector), 1)

    def save(self,):
        pass
