#%%
import numpy as np
import torch
import datasets
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration


class RankT5(nn.Module):
    def __init__(self, base_model="t5-small", rank_token=320089): 
        super(RankT5,self).__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.dense = nn.Linear(512,1) # load and initialize weights
        self.rank_token = rank_token

    def forward(self, input_ids):
        _encoder_hidden_states = self.encoder(input_ids=input_ids)
        encoder_hidden_states = _encoder_hidden_states.last_hidden_state
        _decoder_hidden_states = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        decoder_hidden_states = _decoder_hidden_states.last_hidden_state
        outputs = self.dense(decoder_hidden_states)
        return outputs

#%%
vocab_size = 32_000
input_sentence_size = None
dataset = datasets.load_dataset("ms_marco", "v1.1")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
help(tokenizer)
model = RankT5()
tokenizer.convert_tokens_to_ids("<extra_id_10>")

i = 8
sample = dataset["train"][i]
doc = sample["passages"]["passage_text"][0]
query = sample["query"]
x = f"Query: {query} Document: {doc}"
input_ids = tokenizer(x, return_tensors="pt").input_ids
input_ids >= 32000
score = model(input_ids)

#%% MODEL
