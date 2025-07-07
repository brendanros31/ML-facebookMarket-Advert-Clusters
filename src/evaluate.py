import numpy as np
import pandas as pd

import sklearn.metrics as metrics



# Prediction Stats - KMeans
def evaluate_km(trained_model, X, y):
        labels =  trained_model.labels_

        # Calculating accuracy
        correct_labels = sum(y == labels)
        accuracy = correct_labels / float(y.size)
        print(f'Accuracy: {accuracy:.4f} - {correct_labels} out of {y.size} labelled correct.\n')

        # Silhouette Score
        silhouette = metrics.silhouette_score(X, labels)
        print(f'Silhouette Score: {silhouette:.4f}')



# Prediction Stats - LogReg, RFC
def evaluate(trained_model, X_test, y_test):
        y_pred = trained_model.predict(X_test)   # Predictions
        
        print(trained_model)
        print(f'Classification report: \n{metrics.classification_report(y_test, y_pred)}')
  
