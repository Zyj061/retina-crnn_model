# Convolutional Recurrent Neural Networks for Modeling Retinal Response to Dynamic Natural Scenes

This repository is code of the paper **Unravelling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks.**

`train_off_cnn.py`is script for training CNN model for generated data. You can change the codes to train CRNN models as following:

![train_off_cnn.py](https://github.com/Zyj061/retina-crnn_model/blob/master/off_model.png)

Run the following python script to train and obtain testing results (corresponding to the experiments shown in Fig.2 of our paper):

```
python train_off_cnn 6
```

`test_models.py`is script for testing models on electrophysiological data. Run the following example to test model. We have provided some models that trained on movie2 in the directory `model/movie2/`.

```
python test_models.py --stim movie2 --model crnn_lstm
```

`models.py` and `models_off.py` are codes of the models we have mentioned in the paper. 

`utils.py` and `utils_off.py`, 'visualization.py': codes for testing models or visualizing the hidden units of models

`prune_filters.py`: prune models with spatial autocorrelation or temporal regularity 

`off_data_generator.py` and `data_generator.py` are scripts for preprocessing data that used for training models.The corresponding electrophysiological data can be found in the [link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4ch10) listed in the article, and we have provided the generated data used in Figure 2 of our paper in `data/cell_simpleNL_off_2GC_v3.mat`. You can preprocess the data refer to these files, obtaining the video stimulus input "X" and the corresponding neuron response "r" for training models for electrophysiological data.  

# Citing this Repository

Please cite our work "[Unraveling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S2666389921002051)" when referencing this repository.

# License

The provided implementation is strictly for academic purposes only. Shold you be interested in using our technology for any commercial use, please feel free to contact us.
