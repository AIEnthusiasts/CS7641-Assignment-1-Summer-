import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.datasets import load_digits
import time


digits = load_digits()
X, y = digits.data, digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


models = {
    'Decision Tree': DecisionTreeClassifier(),
    'AdaBoost (Decision Tree)': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME'),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000),
}


param_grids = {
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20],
                      'ccp_alpha': [0.0, 0.01, 0.05, 0.1]},
    'AdaBoost (Decision Tree)': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0],
                                 'estimator__max_depth': [1, 2, 3, 5], 'estimator__ccp_alpha': [0.0, 0.01, 0.05, 0.1]},
    'SVM': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},

    'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']},
}


best_estimators = {}
train_accuracies = []
test_accuracies = []
grid_search_results = {}
training_times = {}

for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    best_estimators[name] = grid_search.best_estimator_


    train_acc = grid_search.best_score_
    train_accuracies.append(train_acc)


    y_pred = best_estimators[name].predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_acc)

    grid_search_results[name] = grid_search.cv_results_
    training_times[name] = end_time - start_time
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Training accuracy for {name}: {train_acc}")
    print(f"Test accuracy for {name}: {test_acc}")
    print(classification_report(y_test, y_pred))



for name, train_acc, test_acc in zip(models.keys(), train_accuracies, test_accuracies):
    print(f"{name} - Training Error Rate: {1 - train_acc:.5f}, Testing Error Rate: {1 - test_acc:.5f}")



for name, results in grid_search_results.items():
    print(f"\nGrid Search results for {name}:")
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"Mean Test Score: {mean_score:.5f}, Parameters: {params}")


for name, train_acc, test_acc in zip(models.keys(), train_accuracies, test_accuracies):
    print(f"{name} - Training Error Rate: {1 - train_acc:.5f}, Testing Error Rate: {1 - test_acc:.5f}")