import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--mode', type=str, default="train", help='train / test')

    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--word-dim', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--val-step', type=int, default=5)
    parser.add_argument('--test-epoch', type=int, default=50)
    parser.add_argument('--start-epoch', type=int, default=0)

    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--train-path', type=str, default='train.npy')
    parser.add_argument('--val-path', type=str, default='val.npy')
    parser.add_argument('--test-path', type=str, default='test.npy')
    parser.add_argument('--glove-path', type=str, default='glove_dict.npy')
    parser.add_argument('--word-dict-path', type=str, default='word_dict.npy')
    parser.add_argument('--word-list-path', type=str, default='word_list.npy')
    parser.add_argument('--model-path', type=str, default="./model")

    args = parser.parse_args()

    return args
