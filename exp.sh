#!/bin/bash

python3 natural_train.py --model SmallFCNet --data-path /home/abhijith/Documents/MSc/research/adver_train/ERAN/data/mnist_train.csv --epochs 2

cd ../ERAN
export GUROBI_HOME="$PWD/gurobi912/linux64"
export PATH="$PATH:${GUROBI_HOME}/bin"
export CPATH="$CPATH:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib

cd tf_verify
rm unverified_mnist.csv
rm verified_mnist.csv
rm attack_mnist.csv
python3 . --netname /home/abhijith/Documents/MSc/research/adver_train/adver_mnist/model/natural/soft/weights/smallfcnet.onnx --epsilon 0.05 --domain deepzono --dataset mnist --num_tests 1000

cd ../../adver_mnist/
python3 csv2attack.py --model SmallFCNet
python3 csv_concat.py
python3 natural_train.py --model SmallFCNet --data-path /home/abhijith/Documents/MSc/research/adver_train/ERAN/tf_verify/final_mnist.csv --epochs 10
