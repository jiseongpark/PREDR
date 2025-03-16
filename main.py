import argparse
from PREDR import Trainer


def parse_args():
    
    parser = argparse.ArgumentParser(description='Training Model Hyperparameters')
    
    parser.add_argument('--model', type=str, default='predr', 
                        choices=['predr', 'dtinet', 'mlp', 'svm'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--chg_dims', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--dg_dims', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--activ', type=str, default='relu')
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--reduction_policy', type=str, default='sum', 
                        choices=['auto', 'none', 'sum', 'sum_over_batch_size'])
    parser.add_argument('--log_path', type=str, default='./Log')
    parser.add_argument('--ckpt_path', type=str, default='./Checkpoint')
    parser.add_argument('--dataset_path', type=str, default='./Dataset')

    return parser.parse_known_args()


if __name__ == '__main__':
    
    args, unparsed = parse_args()
    
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    
    trainer = Trainer(args)
    trainer.train()