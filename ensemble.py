from utils.load_data import load_data
from sklearn.model_selection import train_test_split
import joblib



TARGET = "Class"

MAIN_DF =  load_data("./data/creditcard.csv", TARGET)

train_df, test_df = train_test_split(MAIN_DF, test_size=0.2, random_state=42, stratify=MAIN_DF[TARGET])

rl_model = joblib.load("./models/logistic_regression_model.joblib")
rf_model = joblib.load("./models/random_forest_model.joblib")
nb_model = joblib.load("./models/NB_model.joblib")

X_train, Y_train = [train_df.drop(TARGET, axis=1), train_df[TARGET]]
X_test, Y_test = [test_df.drop(TARGET, axis=1), test_df[TARGET]]

print(rf_model.get_params())

