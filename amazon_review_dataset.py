import argparse
import json
from nltk.tokenize import word_tokenize


def main(args):
    fn = args.data_dir + args.category
    i = 0
    processed = 0
    skipped = 0
    filtered = 0
    with open(fn + '.json', 'r') as f:
        with open(fn + '.tsv', 'w') as f_out:
            while True:
                line = f.readline()
                if not line:
                    break

                if i % 10000 == 0:
                    print(i)
                i += 1

                line = line.strip()
                record = json.loads(line)

                if 'reviewText' not in record or 'overall' not in record:
                    skipped += 1
                    continue

                x = record['reviewText']
                x = word_tokenize(x)
                if len(x) > 100:
                    filtered += 1
                    continue

                x = ' '.join(x)
                y = record['overall']

                f_out.write(str(int(y)) + '\t' + x + '\n')
                processed += 1

    print('skipped:', skipped)
    print('filtered:', filtered)
    print('processed:', processed)
    print('all:', i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/amazon/',
                        help='Data directory.')
    parser.add_argument('--category', type=str, default='Movies_and_TV_5',
                        help='Category name.')
    main(parser.parse_args())
