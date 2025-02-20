from utils.load_data import load_data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib


TARGET = "Class"

MAIN_DF =  load_data("./data/creditcard.csv", TARGET)

train_df, test_df = train_test_split(MAIN_DF, test_size=0.2, random_state=42, stratify=MAIN_DF[TARGET])

param_grid = {  
    'n_estimators': np.arange(20, 200, 20),     # De 100 a 1000, pasos de 100  
    'max_depth': [10, 20, 30, 40, 50],        # Profundidades variadas  
    'min_samples_split': [2, 5, 10, 15],            # Mínimo de muestras para dividir  
    'min_samples_leaf': [1, 2, 4, 6, 8],            # Mínimo de muestras en las hojas  
    'max_features': ['sqrt', 'log2'],               # Número de características  
    'bootstrap': [True, False]                       # Usar bootstrap o no  
}  
random_search = RandomizedSearchCV(  
    RandomForestClassifier(),  
    param_distributions=param_grid,  
    n_iter=100,  # Número de combinaciones a probar  
    scoring=make_scorer(f1_score, pos_label=1),  # Métrica de evaluación (puedes cambiarla)  
    cv=5,  # Tipo de validación cruzada  
    verbose=10,  # Para mostrar los resultados en la consola  
    random_state=42,  # Para reproducibilidad  
    n_jobs=5  # Uso de todos los núcleos disponibles  
)  
random_search.fit(train_df.drop(TARGET, axis=1), train_df[TARGET])
joblib.dump(random_search.best_estimator_, "random_forest_model.joblib")
