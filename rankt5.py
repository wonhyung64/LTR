#%%
import numpy as np
import torch
import datasets

#%%
from module.tokenizer.t5_tokenizer_model import SentencePieceUnigramTokenizer

vocab_size = 32_000
input_sentence_size = None

dataset = datasets.load_dataset("ms_marco", "v1.1")
help(datasets.load_dataset)
help(dataset)

dataset["train"][8]
dataset["train"][10]

#%% MODEL
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

loss = model(input_ids=input_ids, labels=labels).loss

encoder = model.encoder
decoder = model.decoder
