import argparse
from process import Process


# 设定参数列表
def get_args():
    parser = argparse.ArgumentParser(description='Script to train DG_ON_EEG',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tag_id', default=5, help='number of subject')
    parser.add_argument('--seed', default=42, help='number of random seed')
    parser.add_argument('--alpha', default=15e-4, help='hypo parameters 02')
    parser.add_argument('--beta', default=3, help='hypo parameters 01')
    parser.add_argument('--gamma', default=0.3, help='hypo parameters 03')

    parser.add_argument('--windows', default=[[-0.5, 4.0]])

    parser.add_argument('--lr', default=0.01, help='learning rate')
    parser.add_argument('--drop_prob', default=0.5, help='dropout rate')
    parser.add_argument('--batch_size', default=64, help='size of each batch')
    parser.add_argument('--max_epochs', default=1, help='max epochs of running model')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Process(args).running()
