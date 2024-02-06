#%%
'''
"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained with BM25 (only lexical) sampled hard negatives provided by the SentenceTransformers Repo. 

This example has been taken from here with few modifications to train SBERT (MSMARCO-v2) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v2.py) 

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use the negative
passages (the triplets) that are provided by the MS MARCO dataset.

Running this script:
python train_msmarco_v2.py
'''

from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, gzip
import logging
import json


#%%
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download msmarco.zip dataset and unzip the dataset
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Please Note not all datasets contain a dev split, comment out the line if such the case
corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
# _, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
# _, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")


########################################
#### Download MSMARCO Triplets File ####
########################################

# train_batch_size = 75           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
train_batch_size = 2           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^4))

# The triplets file contains 5,028,051 sentence pairs (ref: https://sbert.net/datasets/paraphrases)
triplets_url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/msmarco-query_passage_negative.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-triplets.jsonl.gz")

if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### The triplets file contains tab seperated triplets in each line =>
# 1. train query (text), 2. positive doc (text), 3. hard negative doc (text) 
triplets = []
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        triplet = json.loads(line)
        triplets.append(triplet)

#### Provide any sentence-transformers or HF model
model_name = "distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

#### Prepare triplets samples
train_samples = retriever.load_train_triplets(triplets=triplets)
# train_dataloader = retriever.prepare_train_triplets(train_samples)
train_dataloader = retriever.prepare_train_triplets(train_samples[:1000])
train_samples[1].label
train_samples[1].texts
train_samples[1].guid

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
# #### training SBERT with dot-product
# # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 1
evaluation_steps = 10000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

# %%
retriever.model.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)

#%%
import torch
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import fullname
from tqdm.autonotebook import trange


    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):


train_objectives = [(train_dataloader, train_loss)]
evaluator = ir_evaluator
epochs = num_epochs
steps_per_epoch = None
scheduler: str = 'WarmupLinear'
optimizer_class = torch.optim.AdamW
optimizer_params = {'lr': 2e-5}
weight_decay: float = 0.01
max_grad_norm: float = 1
use_amp = True
show_progress_bar: bool = True

info_loss_functions =  []
for dataloader, loss in train_objectives:
    info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
info_loss_functions = "\n\n".join([text for text in info_loss_functions])

info_fit_parameters = json.dumps({
    "evaluator": fullname(evaluator),
    "epochs": epochs,
    "steps_per_epoch": steps_per_epoch,
    "scheduler": scheduler,
    "warmup_steps": warmup_steps,
    "optimizer_class": str(optimizer_class),
    "optimizer_params": optimizer_params,
    "weight_decay": weight_decay,
    "evaluation_steps": evaluation_steps,
    "max_grad_norm": max_grad_norm,
    }, indent=4, sort_keys=True)

retriever.model._model_card_text = None
retriever.model._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


if use_amp:
    from torch.cuda.amp import autocast
    scaler = torch.cuda.amp.GradScaler()

retriever.model.to(retriever.model._target_device)

dataloaders = [dataloader for dataloader, _ in train_objectives]

# Use smart batching
for dataloader in dataloaders:
    dataloader.collate_fn = retriever.model.smart_batching_collate

loss_models = [loss for _, loss in train_objectives]
for loss_model in loss_models:
    loss_model.to(retriever.model._target_device)

retriever.model.best_score = -9999999

if steps_per_epoch is None or steps_per_epoch == 0:
    steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

num_train_steps = int(steps_per_epoch * epochs)

# Prepare optimizers
optimizers = []
schedulers = []
for loss_model in loss_models:
    param_optimizer = list(loss_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = retriever.model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

    optimizers.append(optimizer)
    schedulers.append(scheduler_obj)


global_step = 0
data_iterators = [iter(dataloader) for dataloader in dataloaders]
# for _ in dataloaders[0]: break
num_train_objectives = len(train_objectives)

skip_scheduler = False
for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):break
    training_steps = 0

    for loss_model in loss_models:
        loss_model.zero_grad()
        loss_model.train()

    for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):break
        for train_idx in range(num_train_objectives):break
            loss_model = loss_models[train_idx]
            optimizer = optimizers[train_idx]
            scheduler = schedulers[train_idx]
            data_iterator = data_iterators[train_idx]

            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloaders[train_idx])
                data_iterators[train_idx] = data_iterator
                data = next(data_iterator)

            features, labels = data
            labels = labels.to(self._target_device)
            features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

            if use_amp:
                with autocast():
                    loss_value = loss_model(features, labels)

                scale_before_step = scaler.get_scale()
                scaler.scale(loss_value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss_value = loss_model(features, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            if not skip_scheduler:
                scheduler.step()

        training_steps += 1
        global_step += 1

        if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

        if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


    self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
    self.save(output_path)

if checkpoint_path is not None:
    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)



triplets[0]


#%%

    def load_train_triplets(self, triplets: List[Tuple[str, str, str]]) -> List[InputExample]:        
        
train_samples = []
retriever.batch_size=2
for idx, start_idx in enumerate(trange(0, len(triplets), retriever.batch_size, desc='Adding Input Examples')):break
    triplets_batch = triplets[start_idx:start_idx+retriever.batch_size]
    for triplet in triplets_batch:
        guid = None
        train_samples.append(InputExample(guid=guid, texts=triplet))

logger.info("Loaded {} training pairs.".format(len(train_samples)))
        return train_samples


class NoDuplicatesDataLoader:

    def __init__(self, train_examples, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        random.shuffle(self.train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.train_examples) / self.batch_size)