# Practica de Regresion Logistica.


## Descripcion del dataset

*El conjunto de datos contiene transacciones realizadas con tarjetas de crédito en septiembre de 2013 por titulares de tarjetas europeos. Este conjunto de datos presenta transacciones que ocurrieron en dos días, donde tenemos 492 fraudes de un total de 284,807 transacciones. El conjunto de datos está altamente desbalanceado, la clase positiva (fraudes) representa el 0.172% de todas las transacciones.*

*Contiene únicamente variables de entrada numéricas que son el resultado de una transformación PCA. Desafortunadamente, debido a problemas de confidencialidad, no podemos proporcionar las características originales ni más información de fondo sobre los datos. Las características V1, V2, … V28 son los componentes principales obtenidos con PCA; las únicas características que no han sido transformadas con PCA son 'Time' y 'Amount'. La característica 'Time' contiene los segundos transcurridos entre cada transacción y la primera transacción en el conjunto de datos. La característica 'Amount' es el monto de la transacción; esta característica puede ser utilizada, por ejemplo, para aprendizaje sensible al costo dependiente del ejemplo. La característica 'Class' es la variable de respuesta y toma el valor 1 en caso de fraude y 0 en caso contrario.*

El objetivo sera crear un meta-modelo de machine learning basandonos en el principio de **stacking** en el contexto del **ensemble learning**, esto para encontrar las mayores precisiones posibles.

Los algoritmos que se planean utilizar seran:

* Regresion Logistica
* SVC
* Random forest

En el caso de SVC solo se implementara en caso de que realice buenas predicciones despues de ser entrenado utilizando un conjunto representativo de los datos, dado el gran tiempo de entrenamiento que conlleva este algoritmo.

Random Forest sera implementado en caso de que el uso intrinseco de PCA en el dataset no altere demasiado las predicciones de este algoritmo.


# Preprocesamiento

1- Manejo de Nans e infinitos: el dataset no tiene nans.

2- Codificacion: el dataset no tiene variables categoricas.

3- Desequilibrio de datos: a pesar del gran desequilibrio de los datos, procederemos en principio sin utilizar ninguna estrategia.

4- Estudio de correlaciones.

No hay correlaciones altas feature-feature.

En el estudio de las correlaciones feature-target, concluimos que las caracteristicas con menos relevancia son las siguientes:

```
['V28', 'Amount', 'V26', 'V25', 'V22', 'V23', 'V15', 'V13', 'V24']
```

Todas las anteriores guardan una correlacion menor al 1% y mayor al -1%

Se contrastaran los resultados anteriores con una estrategia de seleccion de caracteristicas.


5- Seleccion de caracteristicas.

El estudio de seleccion de caracteristicas resulto que las unicas caracteristicas que presentan una relevancia mayor al 1% son:


```
['V17','V14','V12','V16','V10','V11','V9','V18','V4','V7','V3','V26','V21','V1','V8','V5','V19','V2','Time','V20','V6','Amount','V13','V15']
```

## Entrenamiento

1- Busqueda de mejores hiperparametros para Regresion logistica y almacenamiento de modelo en disco: despues de hacer una prueba rapida utilizando el 90% del conjunto de datos inicial, obtuvimos el siguiente rendimiento:


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

Esta es la mejor combinacion de hiperparametros para regresion logistica despues de ejecutar un proceso de seleccion de modelo:

```
{'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'newton-cholesky', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
```

Tener en cuenta que, en el estudio anterior, se utilizaron *todos los ejemplos del dataset*.


### Estudio de hiperparametros de SVC

Para el estudio de hiperparametros optimos de SVC se utilizara el 50% del conjunto original.


