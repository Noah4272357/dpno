# DPNO
This is the code for article [DPNO: A Dual Path Architecture For Neural Operator](https://arxiv.org/abs/2507.12719)

[TOC]
## Structure
To run experiments, you need to set the file path in the /data/dataset.py file and run the main.py file; or you can use the run.sh in /scripts folder.

Below is a detailed intro about each subfolder.
### data
Contains data processing functions.

To run experiment, please change the filepath in /data/dataset.py to your local data path.

Supprted data list
|Name|Dimension|
|---|---|
|Darcy-Flow|2|
|Navier-Stokes|3| 
|Hudgkin-Huxley|1|
|Burgers|1|

### models
Contains different neural operator model. Every file in it contains a get_model function that can return the model corresponding to data name and model name.

### scripts
Contains shell scripts that run experiment automatically.

### Training 
losses.py define the loss function used to train the model
trainer.py define a Trainer class that run the training and validation automatically.

### utils
logging.py defines a Logger class that log the experiments data and save the checkpoints the result will be saved at the /experiments subfolder( if not created, it will create this folder automatically).


