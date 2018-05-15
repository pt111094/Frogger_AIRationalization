# Frogger_AIRationalization

# Installation

Requires Pytorch version 0.3.1 or above

Type in one of the following lines for pytorch installation

- conda install pytorch torchvision -c pytorch
- pip install torch torchvision

Additionally you will also need a compatible version of torchvision(> 0.2.1)

Install torchvision using one of the following commands,

- conda install torchvision -c pytorch
- pip install torchvision
- python setup.py install

# Preprocessing
The requisite vocab file is included in the repo, but incase you need to rebuild the vocabulary run, 

$ python build_vocab.py


# Training
If the vocab file already exists, the run the following command to begin training the model

$ python train_frogger_v2.py

This file includes preprocessing to the image and text data. 

If the data is already available to train the model, simply run the with with the following additional argument, 

$ python train_frogger_v2.py --init_flag True

# Testing 
To test the model choose one of the testing images from './data/FroggerTurkTestingNodiff', and run the following command, 

$ python sample_v2.py --image 'image_location + image_name'
