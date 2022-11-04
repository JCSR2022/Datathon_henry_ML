# Datathon_henry_ML
Datathon henry Lab proyecto individual 2 Machine learning para predictor de precio inmuebles


Objetivo: predecir la categorización de las propiedades entre baratas o caras, considerando como criterio el valor promedio de los precios (la media).​

Actividades:
1. EDA explicado, feature engineerging y, de ser posible, un pipeline con un README acorde, que sirva de introducción al contenido dentro de éste.
2. Script que genere un archivo .csv sólo con las predicciones, teniendo únicamente una sola columna (sin index) que debe llamarse 'pred' y tenga todos los valores de las predicciones, con un valor por fila. De no llamarse así la única columna, no se tomara en cuenta. ​El nombre del archivo debe ser su usuario de GitHub, si su usuario de GitHub es 'JCSR2022', el archivo .csv con las predicciones debe llamarse 'JCSR2022.csv'. 
3. verificar que usuario de GitHub aparezca en el dashboard.

Archivos de entrada: (Se modificaron a parque para que ocupen menos)
'properties_colombia_train.csv': Contiene 197549 registros y 26 dimensiones, el cual incluye la información numérica del precio.
'propiedades_colombia_test.csv': Contiene 65850 registros y 25 dimensiones, el cual no incluye la información del precio.​

Primero se cargaron los datos y se creo una funcion para revisar los faltantes e ir trabajando en ellos

