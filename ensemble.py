from utils.load_data import load_data
from sklearn.model_selection import train_test_split
import joblib
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from utils.precision import precision



TARGET = "Class"

MAIN_DF =  load_data("./data/creditcard.csv", TARGET)

train_df, test_df = train_test_split(MAIN_DF, test_size=0.2, random_state=42, stratify=MAIN_DF[TARGET])

rl_model = joblib.load("./models/logistic_regression_model.joblib")
rf_model = joblib.load("./models/random_forest_model.joblib")
nb_model = joblib.load("./models/NB_model.joblib")

X_train, Y_train = [train_df.drop(TARGET, axis=1), train_df[TARGET]]
X_test, Y_test = [test_df.drop(TARGET, axis=1), test_df[TARGET]]

stacked_X_train, stacked_Y_train = [
    pd.DataFrame({
        "rl" : rl_model.predict(X_train),
        "rf" : rf_model.predict(X_train),
        "nb" : nb_model.predict(X_train)
    }),
    Y_train
]

stacked_X_test, stacked_Y_test = [
    pd.DataFrame({
        "rl" : rl_model.predict(X_test),
        "rf" : rf_model.predict(X_test),
        "nb" : nb_model.predict(X_test)
    }),
    Y_test
]


stacked_model = StackingClassifier(classifiers=[rl_model, rf_model, nb_model],   
                                   meta_classifier=BernoulliNB())

stacked_model.fit(stacked_X_train, stacked_Y_train)

precision(stacked_Y_train, stacked_model.predict(stacked_X_train), "train")
precision(stacked_Y_test, stacked_model.predict(stacked_X_test), "test")
