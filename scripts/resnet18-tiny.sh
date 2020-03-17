CUDA=$1

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 10 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 30 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 50 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 70 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 90 -init_type signed -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 10 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 30 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 50 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 70 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments

python train.py -dataset tiny-imagenet -model resnet18 -sparsity 90 -init_type normal -batch 2048 -cuda $CUDA -lr 0.5 -epochs 50 -wd 0.001 -not_save  -models_dir /home/ezholkovskiy/models/experiments