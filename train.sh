#!/usr/bin/env bash

python main.py --model=vgg11 --linear-layer=TaylorLinear --pre-epochs=0 --post-epochs=160 --pruner=taylor_vgg --compression=1.5 --mask-scope=global --dataset=cifar10 --experiment=singleshot --lr=0.0001
