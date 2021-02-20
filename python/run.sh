#!/bin/bash

for i in $(seq 1 100)
do
	for j in $(seq 1 $i)
	do
		for l in $(seq 1 $j)
		do
			echo "python snn.py --market=1 --labels=/data/agnieszka/clustering/MNIST/origin_data/origin/mnist_train.csv --k=$i --eps=$j --min-pts=$l"
			python snn.py --market=1 --labels=/data/agnieszka/clustering/MNIST/origin_data/origin/mnist_train.csv --k=$i --eps=$j --min-pts=$l
		done
	done
done
