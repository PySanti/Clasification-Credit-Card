from utils.load_data import load_data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from utils.precision import precision


TARGET = "Class"

MAIN_DF =  load_data("./data/creditcard.csv", TARGET)

train_df, test_df = train_test_split(MAIN_DF, test_size=0.2, random_state=42, stratify=MAIN_DF[TARGET])

rl_model = joblib.load("./models/logistic_regression_model.joblib")
rf_model = joblib.load("./models/random_forest_model.joblib")
nb_model = joblib.load("./models/NB_model.joblib")

X_train, Y_train = [train_df.drop(TARGET, axis=1), train_df[TARGET]]
X_test, Y_test = [test_df.drop(TARGET, axis=1), test_df[TARGET]]

print(rf_model.get_params())

print("Modelo: Regresion Logistica")
precision(Y_train, rl_model.predict(X_train), "Train")
precision(Y_test, rl_model.predict(X_test), "Test")

print("Modelo: Random Forest")
precision(Y_train, rf_model.predict(X_train), "Train")
precision(Y_test, rf_model.predict(X_test), "Test")

print("Modelo: Naive Bayes")
precision(Y_train, nb_model.predict(X_train), "Train")
precision(Y_test, nb_model.predict(X_test), "Test")
