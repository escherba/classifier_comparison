import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], header=0, index_col=False)
df.columns = ['label', 'fpr', 'tpr']

plt.figure()
for i, group in df.groupby(df.columns[0]):
    group.plot(x=df.columns[1], y=df.columns[2], title="ROC Curve",
               subplots=True)

if len(sys.argv) <= 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])
