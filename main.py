

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # Allows duplicate libs (safe workaround)
os.environ["OMP_NUM_THREADS"] = "1"                  # Limits OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"
from my_parser import parse_args
from data_loader import load_and_preprocess_data
from my_model_trianer import ModelTrainer
from post_trainer import PostTrainer
import numpy as np


def main():
    seed =[1234,]
    args  = parse_args()
    

   
    model_graph= load_and_preprocess_data(args)
   
    model_trianer = ModelTrainer(args, model_graph)
    model_trianer.train()
    # post_train= PostTrainer(args,model_graph)
    # post_train.train()


main()
