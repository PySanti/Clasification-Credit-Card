from numpy import log
from utils.load_data import load_data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import joblib
import time


TARGET = "Class"

[X_data, Y_data] =  load_data("./data/creditcard.csv", TARGET, use_splitted_dataset=0.5)

svc_param_grid = {  
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],  # gama con algunas opciones numéricas  
    'degree': [2, 3, 4, 5]  # solo necesario si kernel = poly  
}

svc_search = RandomizedSearchCV(SVC(), svc_param_grid, n_iter=100, cv=4, random_state=42, n_jobs=5, verbose=True)
svc_search.fit(X_data, Y_data)

joblib.dump(svc_search.best_estimator_.get_params(), "best_svc_params.joblib")
