# Structure
## data
Contains data processing functions.

To run experiment, please change the filepath in dataset.py to your local data path.

Supprted data list
|Name|Dimension|
|---|---|
|Darcy-Flow|2|
|Navier-Stokes|3| 
|Hudgkin-Huxley|1|
|Burgers|1|
## experiments
Contains experiment logs and checkpoint file.

## models
Contains different neural operator model. Every file in it contains a get_model function that can return the model corresponding to data name and model name.

## scripts
Contains shell scripts that run experiment automatically.




