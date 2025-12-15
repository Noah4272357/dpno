#!/bin/bash
# This script is used to run the training process for different PDEs and modes.
pde_names=(
    'Burgers' 
    'Darcy_Flow'
)

pde_names=(
    'FNO' 
    'DPNO'
)

# noise_names=(
#     'Gaussian'
#     'Laplace'
# )
  
for pde in "${pde_names[@]}"; do  
    for mod_name in "${mod_names[@]}"; do
        echo "Start training $pde data..."  
        python train.py --pde_name $pde --model_name $mod_name 
    done  
done  
# for noise in "${noise_names[@]}"; do
#     for pde in "${pde_names[@]}"; do  
#         for dual in {0..1}; do
#             echo "Start training $pde data..."  
#             python train.py --pde_name $pde --dual_path $dual --noise $noise  
#         done
#     done
# done
  
