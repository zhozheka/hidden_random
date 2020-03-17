CUDA=$1

python train.py -dataset cifar10 -model vgg11bn -sparsity 10 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 30 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 50 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 70 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 90 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 10 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 30 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 50 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 70 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset cifar10 -model vgg11bn -sparsity 90 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.005 -not_save -models_dir /home/ezholkovskiy/models/experiments
