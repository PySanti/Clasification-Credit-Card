# Práctica de clasificación.


## Descripción del dataset

*El conjunto de datos contiene transacciones realizadas con tarjetas de crédito en septiembre de 2013 por titulares de tarjetas europeas. Este conjunto de datos presenta transacciones que ocurrieron en dos días, donde tenemos 492 fraudes de un total de 284,807 transacciones. El conjunto de datos está altamente desbalanceado; la clase positiva (fraudes) representa el 0.172% de todas las transacciones.*

*Contiene únicamente variables de entrada numéricas que son el resultado de una transformación PCA. Desafortunadamente, debido a problemas de confidencialidad, no podemos proporcionar las características originales ni más información de fondo sobre los datos. Las características V1, V2, … V28 son los componentes principales obtenidos con PCA; las únicas características que no han sido transformadas con PCA son 'Time' y 'Amount'. La característica 'Time' contiene los segundos transcurridos entre cada transacción y la primera transacción en el conjunto de datos. La característica 'Amount' es el monto de la transacción; esta característica puede ser utilizada, por ejemplo, para aprendizaje sensible al costo dependiente del ejemplo. La característica 'Class' es la variable de respuesta y toma el valor 1 en caso de fraude y 0 en caso contrario.*

El objetivo será crear un meta-modelo de machine learning basándonos en el principio de **stacking** en el contexto del **ensemble learning**, esto para encontrar las mayores precisiones posibles.

Los algoritmos que se planean utilizar serán:

* Regresión logística
* SVC
* Random forest
* Naive Bayes: Bernoulli

En el caso de SVC solo se implementará en caso de que realice buenas predicciones después de ser entrenado utilizando un conjunto representativo de los datos, dado el gran tiempo de entrenamiento que conlleva este algoritmo.

Random Forest será implementado en caso de que el uso intrínseco de PCA en el dataset no altere demasiado las predicciones de este algoritmo.


# Preprocesamiento

1-. Manejo de Nans e infinitos: el dataset no tiene nans.

2-. Codificación: el dataset no tiene variables categoricas.

3-. Desequilibrio de datos: a pesar del gran desequilibrio de los datos, procederemos en principio sin utilizar ninguna estrategia.

4-. Estudio de correlaciones.

No hay correlaciones altas feature-feature.

En el estudio de las correlaciones feature-target, concluimos que las características con menos relevancia son las siguientes:

```
['V28', 'Amount', 'V26', 'V25', 'V22', 'V23', 'V15', 'V13', 'V24']
```

Todas las anteriores guardan una correlación menor al 1 % y mayor al -1%

Se contrastarán los resultados anteriores con una estrategia de selección de características.


5-. Selección de características.

El estudio de selección de características resultó que las únicas características que presentan una relevancia mayor al 1% son:


```
['V17','V14','V12','V16','V10','V11','V9','V18','V4','V7','V3','V26','V21','V1','V8','V5','V19','V2','Time','V20','V6','Amount','V13','V15']
```

## Entrenamiento

La forma de afrontar el entrenamiento será ejecutando un proceso de selección de modelo para cada uno de los algoritmos mencionados utilizando el 80% del dataset representativo, ignorando el desequilibrio de los datos. Se tratará de lograr la mayor precisión posible utilizando stacking.

*Nota: se concluyó que lo mejor era no utilizar SVC dada la necesidad computacional y su mal rendimiento con conjuntos de datos desequilibrados.*

Resultados en pruebas iniciales con regresion logistica:
```
# hiperparametros

{'solver' : 'newton-cholesky'}

# Resultados

Train set

F1-score, clase positiva: 0.7422003284072249
F1-score, clase negativa : 0.9996166159150792

Test set
F1-score, clase positiva: 0.7333333333333333
F1-score, clase negativa : 0.9996093063233772

```

Esta es la mejor combinación de hiperparámetros para regresión logística después de ejecutar un proceso de selección de modelo.

```
{'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'newton-cholesky', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
```

Una vez almacenado el modelo de regresión logística en disco, se realizo el proceso de selección de modelo para el algoritmo Naive Bayes. Bernoulli, resultado en los siguientes hiperparámetros:

```
{'alpha': np.float64(0.74), 'binarize': 0.0, 'class_prior': None, 'fit_prior': True, 'force_alpha': True}
```

Se procedió a la búsqueda de los mejores hiperparámetros para random forest. Una vez finalizada, se concluyó que los siguientes, son los hiperparámetros más óptimos.

```
{'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': np.int64(180), 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
```


## Evaluación

Después de haber encontrado los hiperparámetros más óptimos para todos los algoritmos mencionados, estos son los resultados individuales de cada modelo.

```
# Modelo: Regresion Logistica

Train

F1-score, clase positiva: 0.7236641221374046
F1-score, clase negativa : 0.9996022284000132

Test

F1-score, clase positiva: 0.7017543859649122
F1-score, clase negativa : 0.9995516601759954


# Modelo: Random Forest

Train

F1-score, clase positiva: 0.987146529562982
F1-score, clase negativa : 0.9999780177265053

Test

F1-score, clase positiva: 0.8804347826086957
F1-score, clase negativa : 0.9998065764023211


# Modelo: Naive Bayes

Train

F1-score, clase positiva: 0.7317073170731707
F1-score, clase negativa : 0.9995890046660059

Test

F1-score, clase positiva: 0.7292817679558011
F1-score, clase negativa : 0.9995692042587236
```

El algoritmo que mejor funciona individualmente es Random Forest, a pesar de estar generando bastante overfitting. Lo anterior seguramente es debido a la normalización por defecto que trae el dataset y por el desequilibrio de los datos. Recordemos que, Random Forest, dada su naturaleza, es un algoritmo que tiende a funcionar peor con conjuntos de datos normalizados y desequilibrados.

Por último, se creará un *meta-modelo* de regresión logística utilizando como base todos los modelos anteriores, basándonos en el concepto de staking dentro del contexto de ensemble learning. Estos son los resultados del meta modelo:

```
Train

F1-score, clase positiva: 0.987146529562982
F1-score, clase negativa : 0.9999780177265053

Test

F1-score, clase positiva: 0.8804347826086957
F1-score, clase negativa : 0.9998065764023211
```

Destacar que el resultado del meta-modelo es indiferente del algoritmo que se utilice en el nivel superior, ya sea RegresionLogistica, Random Forest o Naive Bayes.

Destacar que, utilizando la clase *StackingClassifier* de la libreria *mlxtend*, el meta-modelo es indiferente a hiperparametros optimos. Es decir, para el siguiente codigo:

```
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

```

La logica puede llevar a pensar que se debe hacer algun procedimiento para encontrar los mejores hiperparametros para el *stacked_classifier*, pero la realidad es que, al hacer eso (realizando un procedimiento de seleccion de modelo para BernoulliNB utilizando *stacked_X_train* y *stacked_Y_train*), los resultados empeoran.
