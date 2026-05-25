import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the model training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Meta-training and fine-tuning for KGE models")

    # Dataset configuration
    parser.add_argument('--data_name', type=str, default='hetionet_E',
                        help='Name of the dataset to use')
    parser.add_argument('--state_dir', type=str, default='state',
                        help='dir of save best model ')
   
    parser.add_argument('--benchmark', type=str, default='dataset/new_data',
                        choices=['dataset', 'dataset/new_data'],
                        help='Benchmark type for inductive learning')
    parser.add_argument('--test_type', type=str, default='inference_1',
                        choices=['inference_1', 'inference_2'],
                        help='Type of inference test  for new data')

    # Training mode
    parser.add_argument('--step', type=str, default='meta_train',
                        choices=['meta_train', 'fine_tune'],
                        help='Training step (meta_train or fine_tune)')
    parser.add_argument('--metatrain_state', type=str,
                        default='./state/codex_m_E/codex_m_E.best',
                        help='Path to the pre-trained meta-training state')

    # Subgraph parameters
    parser.add_argument('--num_train_subgraph', type=int, default=5000,
                        help='Number of training subgraphs')
    parser.add_argument('--num_valid_subgraph', type=int, default=200,
                        help='Number of validation subgraphs')
    parser.add_argument('--num_sample_for_estimate_size', type=int, default=50,
                        help='Number of samples for size estimation')
    parser.add_argument('--rw_0', type=int, default=10,
                        help='Random walk parameter 0')
    parser.add_argument('--rw_1', type=int, default=10,
                        help='Random walk parameter 1')
    parser.add_argument('--rw_2', type=int, default=5,
                        help='Random walk parameter 2')
    parser.add_argument('--num_sample_cand', type=int, default=5,
                        help='Number of sample candidates')

    # Meta-training parameters
    parser.add_argument('--num_neg', type=int, default=32,
                        help='Number of negative samples')
    parser.add_argument('--train_num_epoch', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--posttrain_num_epoch', type=int, default=50,
                        help='Number of post-training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--metatrain_check_per_step', type=int, default=314,
                        help='Checkpoint interval during meta-training')
    parser.add_argument('--posttrain_check_per_epoch', type=int, default=314,
                        help='Checkpoint interval during post-training')
    parser.add_argument('--indtest_eval_bs', type=int, default=512,
                        help='Batch size for inductive test evaluation')

    # R-GCN parameters
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of R-GCN layers')
    parser.add_argument('--num_bases', type=int, default=4,
                        help='Number of bases for weight decomposition')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='Embedding dimension')

    # KGE parameters
    parser.add_argument('--kge', type=str, default='TransE',
                        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'],
                        help='Knowledge graph embedding model')
    parser.add_argument('--gamma', type=float, default=10,
                        help='Margin parameter for KGE')
    parser.add_argument('--adv_temp', type=float, default=1,
                        help='Temperature for adversarial sampling')

    # Model graph parameters
    parser.add_argument('--model_graph_type', type=str, default='entity_base',
                        choices=['relation_base', 'entity_base'],
                        help='Type of model graph')
    parser.add_argument('--is_weighted_model_graph', type=bool, default=True,
                        help='Whether to use weighted model graph')
    parser.add_argument('--is_directed_model_graph', type=bool, default=False,
                        help='Whether to use directed model graph')

    # System parameters
    parser.add_argument('--gpu', type=str, default='cuda:0',
                        help='GPU device to use')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility')

    return parser.parse_args()