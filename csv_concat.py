import pandas as pd
import numpy as np
import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
parser.add_argument("--path", default="/home/abhijith/Documents/MSc/research/adver_train/ERAN/tf_verify",
                        help="Model for attack")
args=parser.parse_args()

# Outer path
path=args.path

# Verified path
filepath=os.path.join(path,'verified_mnist.csv')
df1=pd.read_csv(filepath, header=None)
print(len(df1.axes[0]),len(df1.axes[1]))

# Non-verfied path
filepath=os.path.join(path,'attack_mnist.csv')
df2=pd.read_csv(filepath, header=None)
print(len(df2.axes[0]),len(df2.axes[1]))

# Incorrect path
filepath=os.path.join(path,'incorrect_mnist.csv')
df3=pd.read_csv(filepath, header=None)
print(len(df3.axes[0]),len(df3.axes[1]))

# Final path
df=pd.concat([df1,df2])
df=pd.concat([df,df3])
print(len(df.axes[0]),len(df.axes[1]))
filepath=os.path.join(path,'final_mnist.csv')
df.to_csv(filepath,index=False, header=False)
