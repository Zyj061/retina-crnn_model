# Redes Neuronales Convolucionales Recurrentes para Modelar la Respuesta Retiniana a Escenas Naturales Dinámicas

Este repositorio contiene el código del artículo **Unravelling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks.**

`train_off_cnn.py` es un script para entrenar un modelo CNN para datos generados. Puedes cambiar el código para entrenar modelos CRNN de la siguiente manera:

![train_off_cnn.py](https://github.com/Zyj061/retina-crnn_model/blob/master/off_model.png)

Ejecuta el siguiente script de Python para entrenar y obtener resultados de prueba (correspondientes a los experimentos mostrados en la Fig.2 de nuestro artículo):

```
python train_off_cnn 6
```

`test_models.py` es un script para probar modelos en datos electrofisiológicos. Ejecuta el siguiente ejemplo para probar el modelo. Hemos proporcionado algunos modelos entrenados en movie2 en el directorio `model/movie2/`.

```
python test_models.py --stim movie2 --model crnn_lstm
```
También puedes descomentar el siguiente código en `test_models.py` para entrenar el modelo de codificación en movie2.

```
'''
# training model
model = train_model(stim)

# saving the learned model 
make_path(output_path)
save_model(model, output_path) 
'''
```

`models.py` y `models_off.py` son los códigos de los modelos que mencionamos en el artículo.

`utils.py` y `utils_off.py`, `visualization.py`: códigos para probar modelos o visualizar las unidades ocultas de los modelos

`prune_filters.py`: poda modelos con autocorrelación espacial o regularidad temporal

`off_data_generator.py` y `data_generator.py` son scripts para preprocesar datos utilizados para entrenar modelos. Los datos electrofisiológicos correspondientes se pueden encontrar en el [link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4ch10) listado en el artículo, y hemos proporcionado los datos generados utilizados en la Figura 2 de nuestro artículo en `data/cell_simpleNL_off_2GC_v3.mat`. Puedes preprocesar los datos consultando estos archivos, obteniendo la entrada del estímulo de video "X" y la respuesta neuronal correspondiente "r" para entrenar modelos con datos electrofisiológicos.

# Citar este Repositorio

Por favor, cita nuestro trabajo "[Unraveling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S2666389921002051)" cuando hagas referencia a este repositorio.

# Licencia

La implementación proporcionada es estrictamente para fines académicos. Si estás interesado en utilizar nuestra tecnología para cualquier uso comercial, no dudes en contactarnos.
