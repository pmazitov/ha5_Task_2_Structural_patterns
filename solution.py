from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


class StructuralPatternName:
    def __init__(self, classifiers_list) -> None:
#         <YOUR CODE HERE>
        """
        Initialize a class item with a list of classificators
        """
        self.classifiers = classifiers_list


    def fit(self, X_train, y_train):
#         <YOUR CODE HERE>
        """
        Fit classifiers from the initialization stage
        """
        for classifier in self.classifiers:
            classifier.fit(X_train, y_train)
            

    def predict(self, X_test):
#         <YOUR CODE HERE>
        """
        Get predicts from all the classifiers and return
        the most popular answers
        """
        self.predictions = []
        for classifier in self.classifiers:
            self.predictions += [classifier.predict(X_test)]
        self.predictions = np.array(self.predictions)
            
        ans = []
        for i in range(self.predictions.shape[1]):
            preds_for_sample = self.predictions[:, i]
            counts = np.bincount(preds_for_sample)
            ans += [np.argmax(counts)]
        return ans


if __name__ == "__main__":
#     <YOUR CODE HERE>
    """
    1. Load iris dataset
    2. Shuffle data and divide into train / test.
    3. Prepare classifiers to initialize <StructuralPatternName> class.
    4. Train the ensemble
    """
    dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, shuffle=True)
    
    classifiers = [LogisticRegression(), KNeighborsClassifier(), SVC(), DecisionTreeClassifier()]
    my_ensemble = StructuralPatternName(classifiers)
    my_ensemble.fit(X_train, y_train)
    ans = my_ensemble.predict(X_test)
    
    print('Accuracy:', accuracy_score(y_test, ans))
