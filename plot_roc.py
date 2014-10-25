import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import brewer2mpl
from lflearn.metrics import auc as get_auc


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None,
                        help='Where to save the file to')
    parser.add_argument('--input', type=str, default='-',
                        help='input file')
    parser.add_argument('--roc_xmax', type=float, default=1.0,
                        help='X axis maximum for ROC curve')
    parser.add_argument('--title', type=str, default='ROC Curve',
                        help='Plot title')
    namespace = parser.parse_args(args)
    return namespace


def run(args):
    input_file = sys.stdin if args.input == '-' else args.input
    data_frame = pd.read_csv(input_file, header=0, index_col=False)
    data_frame.columns = ['label', 'fpr', 'tpr']
    groups = data_frame.groupby(data_frame.columns[0])

    colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors + \
        brewer2mpl.get_map('Set1', 'qualitative', len(groups) - 8).mpl_colors

    plt.gca().set_color_cycle(colors)
    for i, group in groups:
        xvals = group['fpr']
        yvals = group['tpr']
        auc = get_auc(xvals, yvals)
        plt.plot(xvals, yvals, label="%s (AUC: %0.3f)" % (i, auc))
    plt.title(args.title)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, args.roc_xmax])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)


if __name__ == "__main__":
    run(parse_args())
