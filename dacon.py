#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf


# %%
path = "/Users/wonhyung64/data/Book-Crossing"

df = pd.read_csv(f"{path}/train.csv")

df_features = df.copy()
df_labels = df_features.pop("Book-Rating")

tmp = dict(df_features)
tmp.keys()
tmp["ID"]
i=0
for rate in ratings: 
    print(rate)
    i += 1
    if i == 5: break

ratings = df[["Book-ID", "Book-Title", "Year-Of-Publication", "Publisher", "User-ID", "Age", "Location", "Book-Rating"]]
tfds.features.FeaturesDict({
    "book_id": tf.string,
    "book_title": tf.string,
    "year_of_publication": tf.float32,
    "publisher": tf.string,
    "user_id": tf.string,
    "age": tf.float32,
    "location": tf.string,
    "user_rating": tf.int64,
})

books = df[["Book-ID", "Book-Title", "Year-Of-Publication", "Publisher"]].drop_duplicates(keep="first").reset_index(drop=True)
tfds.features.FeaturesDict({
    "book_id": tf.string,
    "book_title": tf.string,
    "year_of_publication": tf.float32,
    "publisher": tf.string,
})

dict(df)["ID"]
#%%
import itertools

def slices(features):
    for i in itertools.count():
        example = {name: values[i] for name, values in features.items()}
        yield example
features_dict = {name:values[:1] for name, values in df.items()}

for example in slices(df):
    for name, value in example.items():
        print(f"{name}: {value}")
    break

features_df = tf.data.Dataset.from_tensor_slices(df)


yahoo1 = pd.read_table("/Users/wonhyung64/data/Learning to Rank Challenge/ltrc_yahoo/set1.train.txt")
yahoo2 = pd.read_table("/Users/wonhyung64/data/Learning to Rank Challenge/ltrc_yahoo/set2.train.txt")
import tensorflow_datasets as tfds
ds = tfds.load("yahoo_ltrc/set1")
ds = tfds.load("yahoo_ltrc/set2")
ds = tfds.load("mslr_web", data_dir = "/Users/wonhyung64/data/tfds", split="train")
# ds = tfds.load("mslr_web/10k_fold1", split="train")
import tensorflow as tf
ds = ds.map(lambda feature_map: {
    "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
    **feature_map
})
# ds = ds.shuffle(buffer_size=1000).padded_batch(batch_size=32)
ds = ds.padded_batch(batch_size=32)
ds = ds.map(lambda feature_map: (
    feature_map, tf.where(feature_map["_mask"], feature_map.pop("label"), -1.)))
iter(ds)
ds
tmp = iter(ds)
tmp = tfds.as_dataframe(ds["train"].take(10))
a = next(tmp)
a.keys()
a["bm25_anchor"]
a["bm25_body"]
a["bm25_url"]
a["boolean_model_anchor"]
a["boolean_model_body"]
a["boolean_model_url"]

a["doc_id"]
ds.elements.spec[0]
type(a)
a[1]