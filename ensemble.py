from utils.load_data import load_data
from sklearn.model_selection import train_test_split
import joblib
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from utils.precision import precision



TARGET = "Class"

MAIN_DF =  load_data("./data/creditcard.csv", TARGET)

train_df, unseen_df = train_test_split(MAIN_DF, test_size=0.2, random_state=42, stratify=MAIN_DF[TARGET])

rl_model = joblib.load("./models/logistic_regression_model.joblib")
rf_model = joblib.load("./models/random_forest_model.joblib")
nb_model = joblib.load("./models/NB_model.joblib")


validation_df, test_df = train_test_split(unseen_df, test_size=0.3, random_state=42, stratify=unseen_df[TARGET])
X_validation,Y_validation = [validation_df.drop(TARGET, axis=1), validation_df[TARGET]]
X_test, Y_test = [test_df.drop(TARGET, axis=1), test_df[TARGET]]




stacked_X_train, stacked_Y_train = [
    pd.DataFrame({
        "rl" : rl_model.predict(X_validation),
        "rf" : rf_model.predict(X_validation),
        "nb" : nb_model.predict(X_validation)
    }),
    Y_validation
]

stacked_X_test, stacked_Y_test = [
    pd.DataFrame({
        "rl" : rl_model.predict(X_test),
        "rf" : rf_model.predict(X_test),
        "nb" : nb_model.predict(X_test)
    }),
    Y_test
]

stacked_model = StackingClassifier(classifiers=[rl_model, rf_model, nb_model],meta_classifier=BernoulliNB())
stacked_model.fit(stacked_X_train, stacked_Y_train)
joblib.dump(stacked_model, "meta-model.joblib")
