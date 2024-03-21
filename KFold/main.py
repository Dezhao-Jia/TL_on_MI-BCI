import argparse
from process import Process


def get_args():
    parser = argparse.ArgumentParser(description="Script of EEG task classification model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 原始数据信息
    parser.add_argument('--sub_id', default=1, help='number of subject')
    parser.add_argument('--k_fold', default=6, help='number of subject')
    parser.add_argument('--seed', default=2310212233, help='number of random seed')
    parser.add_argument('--if_save', default=False, help='if reverse model information')
    parser.add_argument('--windows', default=[[-0.5, 4.0]])

    # 实验模型部分所需信息
    parser.add_argument('--model_name', default='SB_FCN')
    parser.add_argument('--batch_size', default=16, help='size of each batch')
    parser.add_argument('--lr', default=0.0005, help='learning rate')
    parser.add_argument('--drop_prob', default=0.7, help='dropout rate')
    parser.add_argument('--max_epochs', default=400, help='max epochs of running model')

    return parser.parse_args()


def main():
    args = get_args()
    Process(args).running()


if __name__ == '__main__':
    main()
