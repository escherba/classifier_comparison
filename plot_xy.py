import matplotlib.pyplot as plt
import brewer2mpl
from argparse import ArgumentParser
import json
import os

op = ArgumentParser()
op.add_argument("--x", type=str, required=True,
                help="field to use as X variable")
op.add_argument("--y", type=str, required=True,
                help="field to use as Y variable")
op.add_argument("--xlabel", type=str, required=False,
                help="X-axis label")
op.add_argument("--ylabel", type=str, required=False,
                help="Y-axis label")
op.add_argument("--title", type=str, required=False,
                help="Chart title")
op.add_argument("--xlim", type=float, nargs='+', required=False,
                help="X-axis limits")
op.add_argument("--ylim", type=float, nargs='+', required=False,
                help="Y-axis limits")
op.add_argument("--legend_loc", type=str, required=False, default="best",
                choices=["best", "lower right", "upper right", "lower left",
                         "upper left"],
                help="legend location")
op.add_argument("--save_as", type=str, required=False,
                help="file name to save under")
op.add_argument("--file_labels", type=str, required=False, nargs='+',
                help="file labels (shown in legend)")
op.add_argument("--files", type=str, nargs='+', required=True,
                help="files to plot")

args = op.parse_args()
if args.xlim is not None:
    assert len(args.xlim) == 2
if args.ylim is not None:
    assert len(args.ylim) == 2
if args.file_labels is not None:
    assert len(args.file_labels) == len(args.files)


file_data = []
for f in args.files:
    data = []
    with open(f, 'r') as fh:
        for line in fh:
            try:
                json_obj = json.loads(line)
            except:
                json_obj = None
            if type(json_obj) == dict:
                if args.x in json_obj and args.y in json_obj:
                    data.append((json_obj[args.x], json_obj[args.y]))
    file_data.append((f, data))


AXES_OPTS = dict(
    fontsize=10
)

colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors + \
    brewer2mpl.get_map('Set1', 'qualitative', min(max(len(file_data), 3), 9)) \
    .mpl_colors

plt.figure()
for i, (filename, data) in enumerate(file_data):
    if len(data):
        xs, ys = zip(*data)
        base = os.path.basename(filename)
        label = os.path.splitext(base)[0] \
            if args.file_labels is None \
            else args.file_labels[i]
        plt.plot(xs, ys, label=label, color=colors[i])

plt.legend(loc=args.legend_loc, fontsize=10)
if args.xlabel is None:
    plt.xlabel(args.x, **AXES_OPTS)
else:
    plt.xlabel(args.xlabel, **AXES_OPTS)
if args.ylabel is None:
    plt.ylabel(args.y, **AXES_OPTS)
else:
    plt.ylabel(args.ylabel, **AXES_OPTS)
if args.xlim is not None:
    plt.xlim(*args.xlim)
if args.ylim is not None:
    plt.ylim(*args.ylim)
if args.title is not None:
    plt.title(args.title, fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.grid()

if args.save_as is None:
    plt.show()
else:
    plt.savefig(args.save_as)
