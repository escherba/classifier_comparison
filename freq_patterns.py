import json
import re
import sys
from operator import itemgetter
from os.path import isdir
from itertools import imap
from argparse import ArgumentParser
from lsh_hdc.utils import HTMLNormalizer, RegexTokenizer
from nltk.corpus import stopwords
from fp_growth import find_frequent_itemsets

from utils.lfcorpus import get_data_frames


def read_json(filename):
    with open(filename, 'r') as fh:
        for json_obj in imap(json.loads, fh):
            yield json_obj


def content_from_dir(input_dir, category=None):
    data_train, data_test = get_data_frames(
        input_dir,
        lambda line: json.loads(line))

    data = data_train['data']
    if category is not None:
        filtered = []
        target = data_train['target']
        target_names = data_train['target_names']
        for x, y in zip(data, target):
            if target_names[y] == category:
                filtered.append(x)
    else:
        filtered = data
    for x in transform_data(filtered):
        yield x


def transform_data(it, get_item=None):
    n = HTMLNormalizer()
    t = RegexTokenizer()
    english_stopwords = stopwords.words('english')

    for i, obj in enumerate(it):
        if not i % 1000:
            sys.stderr.write("processed %d lines\n" % i)
        if get_item is not None:
            obj = get_item(obj)
        content = obj.get('content', '')
        result = t.tokenize(n.normalize(content))
        filtered = [w for w in result
                    if w not in english_stopwords
                    and re.match(r'^\d+$', w) is None]
        yield filtered


def content_from_file(filename):
    for x in transform_data(read_json(filename),
                            get_item=itemgetter('object')):
        yield x


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('input', nargs=1)
    p.add_argument('-s', '--minsup', type=int, default=1000,
                   help='Minimum itemset support')
    p.add_argument('-c', '--category', type=str, default="spam",
                   help='Class to perform search on')
    args = p.parse_args()

    input = args.input[0]
    if isdir(input):
        content_gen = content_from_dir(input, category=args.category)
    else:
        content_gen = content_from_file(input)

    for itemset, support in \
            find_frequent_itemsets(content_gen, args.minsup, True):
        print str(support) + ' ' + ' '.join(itemset)
