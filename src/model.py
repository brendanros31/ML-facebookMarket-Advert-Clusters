from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans



def build_model(model_type, params=None):
    if model_type == 'KMeans':
         return KMeans(**(params or {}))
    
    elif model_type == 'LogisticRegression':
         return LogisticRegression(**(params or {})) 
    
    elif model_type == 'RandomForestClassifier':
         return RandomForestClassifier(**(params or {})) 
  


def train_model_KMeans(model, X_train):
    return model.fit(X_train)



def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)
