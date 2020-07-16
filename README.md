# rac-v-satellite-imagery

This project is intended to be used for Unstructured Landing Site Detection for Fixed-Wing UAVs. Using models trained on satellite imagery (which is semantically similar to the expected drone imagery, the algorithms shall be used to find and determine the safest area to land in case of aircraft failure.

### Setting it up

Create the `./data/` folder in the root folder of the repository. Download the dataset from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data into the folder. Unzip the following files:
- grid_sizes.csv
- train_wkt_v4.csv
- sixteen_band
- three_band

### Preprocess data

Run `get_3_band_shapes.py` in the pre_processing folder. This will obtain the shapes for the images to be used for training.

Next run `cache_train.py`. This will extract the requisite training images and labels, and will save them in `train_16.h5` file in `./data/` folder.

### Train the model

Run `main.py` in the train folder. This file may be modified to specify training durations and class to train on. To resume training on a network, `continue_training.py` is provided.

### Postprocessing

The post_processing folder contains the code to make predictions and detect the best road segment to land on. Simply run the `make_predictions.ipynb` notebook, it contains everything necessary for postprocessing. Model weights to use must be specified in the notebook.

## Requirements

A requirements.txt file is included in the root folder of the repository. The file has been tested only for anaconda usage. Using pip to install these packages may overlook some dependencies and is not guaranteed to work immediately. Additional package installations may be required.

Use the file with the following command: `conda install --file requirements.txt`. Conda-forge must be added to the list of channels before running this command.
