
from my_parser import parse
from data_loader import load_and_pre_processing_data
from my_model_trianer import ModelTrainer

def main():
    args  = parse()
   
    model_graph= load_and_pre_processing_data(args)
    
    
    # return
    model_trianer = ModelTrainer(args, model_graph )
    model_trianer.train()


main()
