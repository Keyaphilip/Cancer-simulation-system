import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def train_classifier(X: np.ndarray, y: np.ndarray):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=5)
    clf.fit(X, y)
    return clf, scores.mean(), scores.std()
