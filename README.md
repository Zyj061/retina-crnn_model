# Unravelling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks

This repository is code of the paper **Unravelling neural coding of dynamic natural visual scenes via convolutional recurrent neural networks.**

`train_cnn_lateral_mov3.py` and `train_off_cnn.py` are codes for training CNN model with lateral inhibition neurons for the natural scenes movie 2, and CNN model for generated data. You can change the codes to train CRNN models.

`models.py` and `models_off.py` are codes of the models we have mentioned in the paper. 

`utils.py` and `utils_off.py`, 'visualization.py': codes for testing models or visualizing the hidden units of models

`prune_filters.py`: prune models with spatial autocorrelation or temporal regularity 

