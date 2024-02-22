#%% 
'''
MODULE IMPORT
'''
import pandas as pd
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging

import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.patches import Rectangle

# %%
'''
DATASET LOADING
'''
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

q_keys = [k for k, v in queries.items()]
q_values = [v for k, v in queries.items()]
df = pd.DataFrame(data=zip(q_keys, q_values), columns=["qid", "query"])
df["qid"].nunique()
df["query"].nunique()

q_keys = []
r_nums = []
for k, v in qrels.items():
    q_keys.append(k)
    r_nums.append(len(v))

rel_df = pd.DataFrame(data=zip(q_keys, r_nums), columns=["qid", "rel_num"])
plt.hist(r_nums)

#%%

# dataset
# setting figure
sns.set(rc={"figure.figsize":(8,6)})
plt.rcParams['lines.linewidth'] = 4.0
plt.rcParams['boxplot.flierprops.markersize'] = 10
sns.set_style("white")

# subplot
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

# making yaxis grid
ax1.yaxis.grid()
ax2.yaxis.grid()

#making barplot for each subplot ax1 and ax2
ax1 = sns.histplot(rel_df, ax=ax1)
ax2 = sns.histplot(rel_df, ax=ax2)


plt.xticks(size=16)
plt.yticks(size=16)

# ax1.set_yscale("log")
# ax2.set_yscale("log")

ax1.set_ylim(10000, 500000)
ax2.set_ylim(0, 3000)

ax1.set_yticks([10000, 100000, 300000, 500000])
ax2.set_yticks([1000, 2000, 3000])
fig
ax1.set_xticks([1,2,3,4,5,6,7])
ax2.set_xticks([1,2,3,4,5,6,7])
fig
ax1.set_ylabel("")
ax2.set_ylabel("")

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.text(0., 0.50, "Count", va='center', rotation = 'vertical', fontsize = 16)
ax1.get_xaxis().set_visible(False)

ax1.get_legend().remove()
ax2.get_legend().remove()

labels = ax1.set_yticklabels(['1e+4', '1e+5', '3e+5', '5e+5'], fontsize = 16)
labels = ax2.set_yticklabels(['1000','2000', '3000'], fontsize = 16)
fig
# how big to make the diagonal lines in axes coordinates
d = .7    
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

labels = ax2.set_xticklabels(['$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$'], fontsize = "16")

ax1.set_xlabel("")
ax2.set_xlabel("Number of relevant documents", fontsize = "16")

# %%
'''
EDA
sample query: 554333
sample passage: 16
'''
corpus["16"]["text"]
queries["554333"]
qrels["554333"]
corpus["124237"]["text"]

#%%
corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

q_keys = [k for k, v in queries.items()]
q_values = [v for k, v in queries.items()]
df = pd.DataFrame(data=zip(q_keys, q_values), columns=["qid", "query"])
df["qid"].nunique()
df["query"].nunique()

q_keys = []
r_nums = []
for k, v in qrels.items():
    q_keys.append(k)
    r_nums.append(len(v))

rel_df = pd.DataFrame(data=zip(q_keys, r_nums), columns=["qid", "rel_num"])
plt.hist(r_nums)

#%%

# dataset
# setting figure
sns.set(rc={"figure.figsize":(8,6)})
plt.rcParams['lines.linewidth'] = 4.0
plt.rcParams['boxplot.flierprops.markersize'] = 10
sns.set_style("white")

# subplot
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

# making yaxis grid
ax1.yaxis.grid()
ax2.yaxis.grid()

#making barplot for each subplot ax1 and ax2
ax1 = sns.histplot(rel_df, ax=ax1)
ax2 = sns.histplot(rel_df, ax=ax2)

plt.xticks(size=16)
plt.yticks(size=16)

# ax1.set_yscale("log")
# ax2.set_yscale("log")

ax1.set_ylim(1000, 7000)
ax2.set_ylim(0, 400)

ax1.set_yticks([1000, 3000, 5000, 7000])
ax2.set_yticks([100, 200, 300, 400])

ax1.set_xticks([1,2,3,4])
ax2.set_xticks([1,2,3,4])

ax1.set_ylabel("")
ax2.set_ylabel("")

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.text(0., 0.50, "Count", va='center', rotation = 'vertical', fontsize = 16)
ax1.get_xaxis().set_visible(False)

ax1.get_legend().remove()
ax2.get_legend().remove()

labels = ax1.set_yticklabels(['1000', '3000', '5000', '7000'], fontsize = 16)
labels = ax2.set_yticklabels(['100', '200', '300', '400'], fontsize = 16)
fig
# how big to make the diagonal lines in axes coordinates
d = .7    
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15, linestyle="none", color='k', clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

labels = ax2.set_xticklabels(['$1$', '$2$', '$3$', '$4$'], fontsize = "16")

ax1.set_xlabel("")
ax2.set_xlabel("Number of relevant documents", fontsize = "16")


#%%
from tqdm import tqdm

popularity_dict = {}

for q, pos_d in tqdm(qrels.items()):
    for d in pos_d.keys(): 
        try:
            popularity_dict[d] += 1
        except:
            popularity_dict[d] = 1

x = list(popularity_dict.keys())
y = list(popularity_dict.values())

df_ = pd.DataFrame(data=zip(x,y), columns=["doc_id", "relevance_num"])
df = df_.sort_values("relevance_num", ascending=False).reset_index(drop=True)
plt.bar(df.index[:20000], df["relevance_num"][:20000])





# %%
