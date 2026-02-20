

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # Allows duplicate libs (safe workaround)
os.environ["OMP_NUM_THREADS"] = "1"                  # Limits OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"
from my_parser import parse
from data_loader_en_type import load_and_pre_processing_data
from my_model_trianer import ModelTrainer
from post_trainer import PostTrainer



def main():
    args  = parse()
   
    model_graph= load_and_pre_processing_data(args)
    # return
    model_trianer = ModelTrainer(args, model_graph )
    model_trianer.train()
    # post_train= PostTrainer(args,model_graph)
    # post_train.train()


main()
