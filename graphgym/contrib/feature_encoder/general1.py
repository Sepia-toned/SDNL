import deepsnap
import torch
import torch.nn as nn
from graphgym.config import cfg
from graphgym.register import register_edge_encoder

class LinearEdgeEncoder1(torch.nn.Module):
    def __init__(self, emb_dim: int):
        # emb_dim is not used here.
        super(LinearEdgeEncoder1, self).__init__()
        # For consistency, the edge features will be map to this dimension
        # on the BSI dataset.
        expected_dim = cfg.transaction.feature_amount_dim \
            + cfg.transaction.feature_time_dim
        self.linear = nn.Linear(1, expected_dim)
        cfg.dataset.edge_dim = expected_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        batch.edge_feature = self.linear(batch.edge_feature)
        # 线性层维度转换：1->32
        # 初始的edge_feature为1
        return batch

register_edge_encoder('roland_general1', LinearEdgeEncoder1)