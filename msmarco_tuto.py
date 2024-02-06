#%% 
'''
MODULE IMPORT
'''
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging


# %%
'''
DATASET LOADING
'''
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
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