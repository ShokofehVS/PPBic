import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_style('whitegrid')


# ScatterPlot for Human data
df = pd.read_csv("accuracy_human.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(ax= ax[0], data=df, x=df[df.columns[0]], y=df[df.columns[2]], hue=df[df.columns[3]])
ax[0].set(ylabel="Percentage of Data (%)", xlabel="Liu and Wang Match Score",
          yticks=([1, 5, 10, 30, 50, 70, 90, 100]),
          yticklabels=(['%1', '%5', '%10', '%30', '%50', '%70', '%90', '%100']),
          xticks=([0, 0.25, 0.50, 0.75, 1]))

ax[0].legend(title='Delta', facecolor="white", ncol=2, framealpha=1)

sns.scatterplot(ax= ax[1], data=df, x=df[df.columns[1]], y=df[df.columns[2]], hue=df[df.columns[3]])
ax[1].set(ylabel="Percentage of Data (%)", xlabel="Prelic Relevance Score",
          yticks=([1, 5, 10, 30, 50, 70, 90, 100]),
          yticklabels=(['%1', '%5', '%10', '%30', '%50', '%70', '%90', '%100']),
          xticks=([0, 0.25, 0.50, 0.75, 1.0]))

ax[1].legend(title='Delta', facecolor="white", ncol=2, framealpha=1)


plt.show()
fig.savefig("FunBic_CCS-HumanScores.pdf", dpi=300, transparent=False)

# ScatterPlot for Yeast data
"""df = pd.read_csv("accuracy_yeast.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(ax= ax[0], data=df, x=df[df.columns[0]], y=df[df.columns[2]], hue=df[df.columns[3]])
ax[0].set(ylabel="Percentage of Data (%)", xlabel="Liu and Wang Match Score",
          yticks=([1, 5, 10, 30, 50, 70, 90, 100]),
          yticklabels=(['%1', '%5', '%10', '%30', '%50', '%70', '%90', '%100']),
          xticks=([0, 0.25, 0.50, 0.75, 1]))

ax[0].legend(title='Delta', facecolor="white", ncol=2, framealpha=1)

sns.scatterplot(ax= ax[1], data=df, x=df[df.columns[1]], y=df[df.columns[2]], hue=df[df.columns[3]])
ax[1].set(ylabel="Percentage of Data (%)", xlabel="Prelic Relevance Score",
          yticks=([1, 5, 10, 30, 50, 70, 90, 100]),
          yticklabels=(['%1', '%5', '%10', '%30', '%50', '%70', '%90', '%100']),
          xticks=([0, 0.25, 0.50, 0.75, 1.0]))

ax[1].legend(title='Delta', facecolor="white", ncol=2, framealpha=1)


plt.show()
fig.savefig("FunBic_CCS-YeastScores.pdf", dpi=300, transparent=False)"""