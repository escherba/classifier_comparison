import matplotlib.pyplot as plt
import pandas as pd
import sys
import brewer2mpl

df = pd.read_csv(sys.argv[1], header=0, index_col=False)
df.columns = ['label', 'fpr', 'tpr']

plt.figure()
groups = df.groupby(df.columns[0])
colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors + \
    brewer2mpl.get_map('Set1', 'qualitative', len(groups) - 8).mpl_colors

for idx, (i, group) in enumerate(groups):
    group.plot(x=df.columns[1], y=df.columns[2], label=str(i),
               title="ROC Curve", color=colors[idx], subplots=True)

plt.legend(loc='best')
plt.xlim([0.0, 0.10])

if len(sys.argv) <= 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])
