import json
import re
import sys
from itertools import imap
from argparse import ArgumentParser
from lsh.utils import HTMLNormalizer, RegexTokenizer
from nltk.corpus import stopwords
from fp_growth import find_frequent_itemsets


def get_content(filename):
    english_stopwords = stopwords.words('english')
    n = HTMLNormalizer()
    t = RegexTokenizer()

    with open(filename, 'r') as fh:
        for i, json_obj in enumerate(imap(json.loads, fh)):
            if not i % 1000:
                sys.stderr.write("processed %d lines\n" % i)
            content = json_obj['object'].get('content', '')
            result = t.tokenize(n.normalize(content))
            filtered = [w for w in result
                        if w not in english_stopwords
                        and re.match(r'^\d+$', w) is None]
            yield filtered


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('infile', nargs=1)
    p.add_argument('-s', '--minsup', type=int, default=274,
                   help='Minimum itemset support')
    args = p.parse_args()

    content_gen = get_content(args.infile[0])
    for itemset, support in find_frequent_itemsets(content_gen,
                                                   args.minsup, True):
        print str(support) + ' ' + ' '.join(itemset)
