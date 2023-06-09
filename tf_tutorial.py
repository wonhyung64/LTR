#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
from typing import Dict, Tuple


#%%
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})

movies = movies.map(lambda x: x["movie_title"])
users = ratings.map(lambda x: x["user_id"])

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(users.batch(1000))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies.batch(1000))

key_func = lambda x: user_ids_vocabulary(x["user_id"])
reduce_func = lambda key, dataset: dataset.batch(100)
ds_train = ratings.group_by_window(
    key_func=key_func, reduce_func=reduce_func, window_size=100
)

#%%
for x in ds_train.take(1):
    for key, value in x.items():
        print(f"Shape of {key}: {value.shape}")
        print(f"Example values of {key}: {value[:5].numpy()}")
        print()

#%%
def _features_and_labels(x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    labels = x.pop("user_rating")
    return x, labels

ds_train = ds_train.map(_features_and_labels)

ds_train = ds_train.apply(
    tf.data.experimental.dense_to_ragged_batch(batch_size=32)
)

for x, label in ds_train.take(1):
    for key, value in x.items():
        print(f"Shape of {key}: {value.shape}")
        print(f"Example values of {key}: {value[:3, :3].numpy()}")
        print()
    print(f"Shape of label: {label.shape}")
    print(f"Example values of label: {label[:3, :3].numpy()}")

user_ids_vocabulary.vocabulary_size()
movie_titles_vocabulary.vocabulary_size()


#%%
class MovieLensRankingModel(tf.keras.Model):
    def __init__(self, user_vocab, movie_vocab):
        super().__init__()
        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
        self.user_embed = tf.keras.layers.Embedding(user_vocab.vocabulary_size(), 64)
        self.movie_embed = tf.keras.layers.Embedding(movie_vocab.vocabulary_size(), 64)

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_embed(self.user_vocab(features["user_id"]))
        movie_embeddings = self.movie_embed(self.movie_vocab(features["movie_title"]))
        return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)


#%%
model = MovieLensRankingModel(user_ids_vocabulary, movie_titles_vocabulary)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss = tfr.losses._softmax_loss(name="loss/softmax", reduction=True)
eval_metrics = [
    tfr.metrics._NDCGMetric(name="metric/ndcg", topn=10),
    tfr.metrics._MRRMetric(name="metric/mrr"),
]



from tqdm import tqdm

epochs = 3
for epoch in range(epochs):
    progress_bar = tqdm(enumerate(ds_train))
    # progress_bar.set_description(f"Epoch {epoch}/{epochs}: ")
    for _, (features, ratings)  in progress_bar:
        b = ratings.shape[0]
        with tf.GradientTape() as tape:
            pred = model(features)
            loss = - tf.reduce_sum(ratings * tf.math.log(tf.nn.softmax(pred, axis=-1))) / b
        progress_bar.set_description(f"Epoch: {epoch}/{epochs} -  loss: {round(loss.numpy(), 3)}")
        grads = tape.gradient(loss, model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        progress_bar.set_description(f"Epoch: {epoch}/{epochs} -  loss: {round(loss, 3)}")



#%%
import json
file = "/Users/wonhyung64/Downloads/ko_alpaca_style_dataset.json"
file = "/Users/wonhyung64/Downloads/small.json"
with open(file, "r") as f:
    doc = json.load(f)
len(doc)

import os
os.listdir("~/Users/wonhyung64/Downloads")
tmp = doc[:100]
with open("/Users/wonhyung64/Downloads/small.json", "w") as f:
    json.dump(tmp, f)