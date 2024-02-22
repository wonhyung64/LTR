#%%
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function


class InvPrefExplicit(nn.Module):
    def __init__(
            self, user_num: int, item_num: int, env_num: int, factor_num: int, reg_only_embed: bool = False,
            reg_env_embed: bool = True
    ):
        super(InvPrefExplicit, self).__init__()
        self.user_num=user_num
        self.item_num=item_num
        self.env_num=env_num

        self.factor_num: int = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)

        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim=1)
        env_aware_mid_score: torch.Tensor = torch.sum(env_aware_preferences, dim=1)
        env_aware_score: torch.Tensor = invariant_score + env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)


class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# %%
data_dir = "./data/coat"

os.listdir(f"{data_dir}/user_item_features")

train = np.genfromtxt(f'{data_dir}/train.ascii', encoding='ascii')
test = np.genfromtxt(f'{data_dir}/test.ascii', encoding='ascii')
propensities = np.genfromtxt(f'{data_dir}/propensities.ascii', encoding='ascii')
user_feat = np.genfromtxt(f'{data_dir}/user_item_features/user_features.ascii', encoding='ascii')
item_feat = np.genfromtxt(f'{data_dir}/user_item_features/item_features.ascii', encoding='ascii')


with open(f'{data_dir}/user_item_features/user_features_map.txt', "r") as f:
    user_feat_map = f.readlines()
    
with open(f'{data_dir}/user_item_features/item_features_map.txt', "r") as f:
    item_feat_map = f.readlines()


#%%
item_num = item_feat.shape[0]
user_num = user_feat.shape[0]
env_num = 4
factor_num = 30 #embedding dimesion
reg_only_embed = True
reg_env_embed = False

batch_size = 1024
epochs = 1000
cluster_interval = 30
evaluate_interval = 10
lr = 0.01

    # "invariant_coe": 2.050646960185343,
    # "env_aware_coe": 8.632289952059462,
    # "env_coe": 5.100067503854663,
    # "L2_coe": 7.731619515414727,
    # "L1_coe": 0.0015415961377493945,
    # "alpha": 1.7379692382330174,
    # "use_class_re_weight": True,
    # "use_recommend_re_weight": True,
    # 'test_begin_epoch': 0,
    # 'begin_cluster_epoch': None,
    # 'stop_cluster_epoch': None,

# EVALUATE_CONFIG: dict = {
    # 'eval_metric': 'mse'
# }

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]
# RANDOM_SEED_LIST = [17373331, 17373522, 17373507, 17373511, 17373423]
# RANDOM_SEED_LIST = [17373331]
# RANDOM_SEED_LIST = [999]


model = InvPrefExplicit(
    user_num, item_num, env_num, factor_num, reg_only_embed, reg_env_embed
    )

#%%
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

train_data_path = f"{data_dir}/train.csv"
test_data_path = f"{data_dir}/test.csv"

import pandas as pd
train_df: pd.DataFrame = pd.read_csv(train_data_path)  # [0: 100000]
test_df: pd.DataFrame = pd.read_csv(test_data_path)

_train_data: np.array = train_df.values.astype(np.int64)
_test_data: np.array = test_df.values.astype(np.int64)

_train_data_tensor: torch.Tensor = torch.LongTensor(_train_data).to(device)
_test_data_tensor: torch.Tensor = torch.LongTensor(_test_data).to(device)

user_positive_interaction = []

_user_num = int(np.max(_train_data[:, 0].reshape(-1))) + 1
_item_num = int(np.max(_train_data[:, 1].reshape(-1))) + 1

_train_pairs: np.array = _train_data[:, 0:2].astype(np.int64).reshape(-1, 2) # user_id, item_id
_test_pairs: np.array = _test_data[:, 0:2].astype(np.int64).reshape(-1, 2)

_train_pairs_tensor: torch.Tensor = torch.LongTensor(_train_pairs).to(device)
_test_pairs_tensor: torch.Tensor = torch.LongTensor(_test_pairs).to(device)

_train_scores: np.array = _train_data[:, 2].astype(np.float64).reshape(-1)
_test_scores: np.array = _test_data[:, 2].astype(np.float64).reshape(-1)

_train_scores_tensor: torch.Tensor = torch.Tensor(_train_scores).to(device)
_test_scores_tensor: torch.Tensor = torch.Tensor(_test_scores).to(device)

