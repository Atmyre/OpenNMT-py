import argparse
import os

TRAIN_RAW_FOLDER = 'training-monolingual.tokenized.shuffled'
TRAIN_FILENAME = 'train.txt'
TEST_FILENAME = 'test.txt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oneb_folder', type=str, required=True,
                        help='folder of oneb dataset')
    parser.add_argument('--maxlen', type=int, default=15,
                        help='max number of tokens in a sentence to take')
    parser.add_argument('--train_size', type=int, default=1000*1000,
                        help='sent count for train')
    parser.add_argument('--test_size', type=int, default=3*1000,
                        help='sent count for test')
    parser.add_argument('--out_folder', type=str, default='data_oneb',
                        help='folder to dump train/test datasets')
    args = parser.parse_args()

    return args


def iget_next_filepath(folder):
    for root, directories, files in os.walk(folder):
        for file_ in files:
            yield os.path.join(root, file_)


def iget_next_line(folder):
    train_foldername = os.path.join(folder, TRAIN_RAW_FOLDER)
    for filepath in iget_next_filepath(train_foldername):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()


def iget_filtered_line(folder, maxlen):
    for line in iget_next_line(folder):
        tokens_count = len(line.split(' '))
        if tokens_count > maxlen:
            continue
        yield line


def dump_lines(line_getter, out_filepath, size):
    with open(out_filepath, 'w', encoding='utf-8') as f:
        for idx, line in enumerate(line_getter, start=1):
            f.write(line + '\n')
            if idx == size:
                break
    print('Dump lines into: {}'.format(out_filepath))


def create_train_test_sets(inp_folder, train_fp, test_fp, train_size, test_size,
                           maxlen):
    line_getter = iget_filtered_line(inp_folder, maxlen)
    dump_lines(line_getter, train_fp, train_size)
    dump_lines(line_getter, test_fp, test_size)


def main(args):
    inp_folder = args.oneb_folder
    train_fp = os.path.join(args.out_folder, TRAIN_FILENAME)
    test_fp = os.path.join(args.out_folder, TEST_FILENAME)

    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)

    create_train_test_sets(inp_folder, train_fp, test_fp, args.train_size, args.test_size,
                           args.maxlen)


if __name__ == '__main__':
    args = parse_args()
    main(args)

