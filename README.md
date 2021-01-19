# Reconocimiento multimodal de emociones con datos faltantes
Repositorio con él código para la reproducibilidad del proyecto.

Todos los conjuntos de datos aquí usados son de acceso libre para investigación pero se requiere solicitar aceso a ellos, por este motivo y el tamaño de estos conjuntos no se suben los datos a este repositorio pero se brindan enlaces para acceder a ellos.

Los conjuntos de datos usados son:

## IEMOCAP (modaldiad AUDIO):
Se puede solicitar el acceso a este conjunto de datos para investigación en: https://sail.usc.edu/iemocap/

## HCI-TAGGING (modaldiad VIDEO):
Se puede solicitar el acceso a este conjunto de datos para investigación en: https://mahnob-db.eu/hci-tagging/

## AROUSAL VALENCE FB POSTS (modalidad TEXTO):
Este conjunto de datos peude ser descargado de: http://mypersonality.org/wiki/doku.php?id=download_databases

El orden del proyecto es el sigueinte:

-   Se descargan los datos.
-   Se llevan a un formato estándar para el proyecto (proceso automatizado mediante scripts).
-   Se crean los modelos unimodales para cada modaldiad independientemente.
-   Se fusionan los resultados de cada modalidad para crear el conjunto de datos artificial.
-   Se crean los modelos apra los tres métodos propuestos.
-   Se crean los modelos de predicción de errores para mejorar los resultados de los métodos.

Para al ejecución del proyecto se requiere Python 3 y varias librerias. El orden de ejecución es el siguiente:

-   Una vez descargados los datos, se ejecuta el script: metadata_[identificador del conjunto de datos].py para cada uno de los tres conjuntos de datos aquí utilizados.
-   Se ejecuta el main.py presente en la carpeta de cada modalidad (audio, face y text).
-   Se usa el script mix_labels_2.py para fusionar los datos resutlantes de cada modaldiad y así crear el conjunto de datos artificial.
-   Cada uno de los tres metodos propuestos (flf_nn, dlf_zero_padding y dlf_rnn_dinamyc_input) y el modelo base (base_model) tiene un script "naive" para ejecutar el método sin encargarse de los datos faltantes y un script apra ejecutar el método teniendo en cuenta la disponibildiad de modalidades en todo momento. Los tres métodos propuestos poseen un script "errors" el cual crea modelos de predicción de errores y suma estas predicciones a los métodos para mejorar sus resultados. todos los scripts anteriores se peude ejecutar mediante el comando: python3 [nombre del script].py