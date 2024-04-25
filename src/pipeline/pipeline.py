import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import Classifier
from feature import Feature


class Pipeline:
    def __init__(self, data: pd.DataFrame, features: list[Feature]):
        assert isinstance(data, pd.DataFrame)
        for elem in features:
            assert issubclass(elem, Feature)

        self._data, self._labels = self._transform(data=data)
        self._features: np.ndarray = self._calculate_features(features=features)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._features, self._labels, test_size=0.2
        )
        self.X_train = self.X_train.reshape(len(self.X_train),1)
        self.X_test = self.X_test.reshape(len(self.X_test),1)

    def train_classifier(self, classifier_reference: Classifier) -> Classifier:
        classifier: Classifier = classifier_reference()
        classifier.train(X_train=self.X_train, y_train=self.y_train)
        predicted_labels = classifier.predict(X_test=self.X_test)
        classifier.evaluate(y_true=self.y_test, y_pred=predicted_labels)

        return classifier

    def infer(data: pd.DataFrame, classifier: Classifier):
        """Use this method to predict labels on held-out data"""
        assert isinstance(data, pd.DataFrame)

    def _calculate_features(self, features: list[Feature]) -> np.ndarray:
        feature_vector = np.array([])
        
        for feature_representation in features:
            feature: Feature = feature_representation()
            feature_vector = np.append(feature_vector, feature.process(data = self._data))
        
        return np.array(feature_vector)

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop('target', axis=1), data.iloc[:,-1:].values
