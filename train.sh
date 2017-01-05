#!/usr/bin/env bash

for i in 1 2 3 4
do
    python2.7 -u mnist.py --beta 100 --scale 0.99 --num-epoch 20 --train --test --op-impl cpp --gpu 0 --margin $i
done
