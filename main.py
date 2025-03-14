
from my_parser import parse
from data_loader import load_and_pre_processing_data
from model_trainer import Model_trainer

def main():
    args  = parse()
   
    model_graph= load_and_pre_processing_data(args)
    
    # return
    model_trianer = Model_trainer(args, model_graph )
    model_trianer.train()


main()
