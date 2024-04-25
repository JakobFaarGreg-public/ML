import pandas as pd
from pipeline import Pipeline
from classifier import SVM
from feature import Sum
from sklearn.datasets import load_iris


def main():
    # Get data
    iris = load_iris(as_frame=True)
    data = iris.data
    data = data.join(iris.target)
    print(data)
    
    # ETL
    p = Pipeline(data=data, features=[Sum])
    
    # Fit + predict
    clf = p.train_classifier(SVM)
    
    # Infer on new data
    # p.infer(data=data, classifier=clf)

if __name__ == "__main__":
    main()