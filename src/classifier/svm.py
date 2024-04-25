import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from classifier import Classifier


class SVM(Classifier):
    def __init__(self):
        self.classifier = svm.SVC()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.classifier.fit(X=X_train, y=y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X_test)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        print(f1_score(y_true=y_true, y_pred=y_pred, average="weighted"))
