CUDA=$1

python train.py -dataset bowl -model unet -sparsity 10 -init_type normal -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 30 -init_type normal -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 50 -init_type normal -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 70 -init_type normal -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 90 -init_type normal -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 10 -init_type signed -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 30 -init_type signed -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 50 -init_type signed -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 70 -init_type signed -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset bowl -model unet -sparsity 90 -init_type signed -batch 16 -cuda $CUDA -lr 0.1 -epochs 50 -wd 0.0005 -not_save -models_dir /home/ezholkovskiy/models/experiments
