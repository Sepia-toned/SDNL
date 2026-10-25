import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from deepsnap.graph import Graph

from graphgym.config import cfg
from graphgym.models.head import head_dict
from graphgym.models.layer import (GeneralLayer, GeneralMultiLayer,
                                   GeneralRecurrentLayer,GeneralRecurrentLayer1,GRUGraphRecurrentLayer,
                                   GraphRecurrentLayerWrapper,
                                   BatchNorm1dNode, BatchNorm1dEdge,Linear,
                                   layer_dict)
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.init import init_weights
from graphgym.models.feature_encoder import node_encoder_dict, \
    edge_encoder_dict

from graphgym.contrib.stage import *
import graphgym.register as register
from graphgym.register import register_network

########### Layer ############
# Methods to construct layers.
recurrent_layer_types = [
    'dcrnn', 'evolve_gcn_o', 'evolve_gcn_h', 'gconv_gru', 'gconv_lstm',
    'gconv_lstm_baseline', 'tgcn', 'edge_conv_gru'
]

def GNNLayer(dim_in, dim_out, has_act=True, id=0):
    if cfg.gnn.layer_type in recurrent_layer_types:
        # For baseline Recurrent GNNs, instantiate directly.
        return layer_dict[cfg.gnn.layer_type](dim_in, dim_out, id=id)
    else:
        # In ROLAND case, call the GeneralRecurrentLayer wrapper layer to
        # convert cfg.gnn.layer_type object to a recurrent layer that updates
        # batch.node_states as well.
        if cfg.gnn.embed_update_method == 'moving_average':
            # The original implementation.
            # TODO: use an unified implementation.
            return GeneralRecurrentLayer(cfg.gnn.layer_type, dim_in, dim_out,
                                         has_act, id=id)
        else:
            return GraphRecurrentLayerWrapper(cfg.gnn.layer_type, dim_in, dim_out,
                                              has_act, id=id)

def GNNLayer1(dim_in, dim_out, has_act=True, id=0):
    if cfg.gnn.layer_type in recurrent_layer_types:
        # For baseline Recurrent GNNs, instantiate directly.
        return layer_dict[cfg.gnn.layer_type](dim_in, dim_out, id=id)
    else:
        # In ROLAND case, call the GeneralRecurrentLayer wrapper layer to
        # convert cfg.gnn.layer_type object to a recurrent layer that updates
        # batch.node_states as well.
        '''if cfg.gnn.embed_update_method == 'moving_average':
            # The original implementation.
            # TODO: use an unified implementation.
            return GeneralRecurrentLayer1(cfg.gnn.layer_type, dim_in, dim_out,
                                         has_act, id=id)
        else:
            return GraphRecurrentLayerWrapper(cfg.gnn.layer_type, dim_in, dim_out,
                                              has_act, id=id)'''
        return GeneralRecurrentLayer1(cfg.gnn.layer_type, dim_in, dim_out,
                                         has_act, id=id)

def GNNPreMP(dim_in, dim_out):
    r"""Constructs preprocessing layers: dim_in --> dim_out --> dim_out --> ... --> dim_out"""
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out,
                             final_act=True)

########### Block: multiple layers ############

class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipBlock, self).__init__()
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, dim_out, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = \
                torch.cat((node_feature, self.f(batch).node_feature), 1)
        else:
            raise ValueError(
                'cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch

# ————————————————————————Attention模块———————————————————————————————————#

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, nb_nodes, hid_units, num_heads):
        super(MultiHeadAttentionModule, self).__init__()
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.num_heads = num_heads
        self.heads = nn.ModuleList([att(nb_nodes, hid_units) for _ in range(num_heads)])
        
        self.linear = nn.Linear(64, 64)
        self.prelu = nn.PReLU()
        
    def forward(self, batch, batch1):
        temp_list=[]
        for idx in range(self.num_heads):
            temp,nodes= self.heads[idx](batch,batch1)
            temp_list.append(temp)

        temp1=torch.mean(torch.cat(temp_list, 0), 0)
        
        '''temp1=F.normalize(temp1, p=2, dim=-1)
        temp1=self.prelu(temp1)'''
        
        # batch.node_feature[nodes]=temp1
        batch.node_feature = 0.5*(batch.node_feature+batch1.node_feature)

        '''batch.node_feature=self.linear(batch.node_feature)
        batch.node_feature=F.normalize(batch.node_feature, p=2, dim=-1)
        batch.node_feature=self.prelu(batch.node_feature)'''

        return batch

class att(nn.Module):
    def __init__(self,nb_nodes,hid_units):
        super(att, self).__init__()
        self.A = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(2)])
        self.nb_nodes=nb_nodes
        self.hid_units=hid_units
        self.weight_init()

    def weight_init(self):
        for i in range(2):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, batch,batch1):
        feat_all = []
        nodes = list(set(batch.edge_index.view(-1).tolist()))
        num_nodes=len(nodes)
        feature=batch.node_feature[nodes]
        feature1=batch1.node_feature[nodes]

        feat_all.append(feature)
        feat_all.append(feature1)
        feat_after, feat_after_attn = self.attn_feature(feat_all,num_nodes) # 注意力机制

        return feat_after,nodes
    
    def attn_feature(self, features,num_nodes):
        features_attn = []
        for i in range(2):
            features_attn.append((self.A[i](features[i])))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features,0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(2, num_nodes, self.hid_units).sum(0).unsqueeze(0)

        return features, features_attn

########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    r"""Simple Stage that stacks GNN layers"""

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, id=i)
            self.add_module('layer{}'.format(i), layer)
        
        # self.attention_module = AttentionModule(nb_nodes=38918,hid_units=64) # Attention applied to input features
        # self.crosslayer=CrossLayer(L1_dim=64,L2_dim=64,bias=True, bet_weight=True)

        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch

class GNNStackStage1(nn.Module):
    r"""Simple Stage that stacks GNN layers"""

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage1, self).__init__()
        

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer1(d_in, dim_out, id=i)
            self.add_module('layer{}'.format(i), layer)
        
        # self.attention_module = AttentionModule(nb_nodes=38918,hid_units=64) # Attention applied to input features
        # self.crosslayer=CrossLayer(L1_dim=64,L2_dim=64,bias=True, bet_weight=True)

        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch

stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
    'stack1': GNNStackStage1
}

stage_dict = {**register.stage_dict, **stage_dict}

############# Network Reconstruction Module ###################
class CrossLayer(nn.Module):
    def __init__(self, L1_dim, L2_dim, bias=None, bet_weight=True):
        super(CrossLayer, self).__init__()
        self.L1_dim = L1_dim
        self.L2_dim = L2_dim
        self.bet_weight = bet_weight
        self.weight = Parameter(torch.FloatTensor(L1_dim, L2_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(L2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):   
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, sub_feature,sub_feature1):
        L1_features=sub_feature
        L2_features=sub_feature1
        if self.bet_weight:
            temp = torch.mm(L1_features, self.weight)
            output = torch.mm(temp, torch.t(L2_features))
            '''if self.bias is not None:
                output = output + self.bias'''            
        else:
            output = torch.mm(L1_features, torch.t(L2_features))
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.L1_dim) + ' -> ' \
               + str(self.L2_dim) + ')'

class CCN(nn.Module):
    def __init__(self):
        super(CCN, self).__init__()
        self.crosslayer=CrossLayer(L1_dim=64, L2_dim=64, bias=True, bet_weight=True)

    def forward(self,batch,batch1):

        # 提取子图特征
        all_nodes_batch = torch.unique(torch.cat((batch.edge_index[0], batch.edge_index[1])))
        sub_list = torch.sort(all_nodes_batch).values
        sub_feature = batch.node_feature[sub_list]

        all_nodes_batch1 = torch.unique(torch.cat((batch1.edge_index[0], batch1.edge_index[1])))
        sub_list1 = torch.sort(all_nodes_batch1).values
        sub_feature1 = batch1.node_feature[sub_list1]

        # 将张量转换为 Python 集合
        sub_list_set = set(sub_list.tolist())
        sub_list1_set = set(sub_list1.tolist())

        temp1 = torch.mm(sub_feature, torch.t(sub_feature)).detach().clone()
        temp2 = torch.mm(sub_feature1, torch.t(sub_feature1)).detach().clone()
        temp3 = self.crosslayer(sub_feature,sub_feature1).detach().clone()

        key1='recon_dy'
        key2='recon_st'
        key3='recon_bt'
        
        # Check if the attributes exist in batch
        if hasattr(batch, key1):
            batch.recon_dy = temp1
        else:
            setattr(batch, key1, temp1)

        if hasattr(batch, key2):
            batch.recon_st = temp2
        else:
            setattr(batch, key2, temp2)

        if hasattr(batch, key3):
            batch.recon_bt = temp3
        else:
            setattr(batch, key3, temp3)

        key4='batch1_edge'
        temp4=batch1.edge_index.detach().clone()

        if hasattr(batch,key4):
            batch.batch1_edge = temp4
        else:
            setattr(batch,key4,temp4)
        
        key5='batch_list'
        temp5=sub_list

        key6='batch1_list'
        temp6=sub_list1

        if hasattr(batch,key5):
            batch.batch_list = temp5
        else:
            setattr(batch,key5,temp5)

        if hasattr(batch,key6):
            batch.batch1_list = temp6
        else:
            setattr(batch,key6,temp6)

        return batch

########### Model: start + stage + head ############
    
import copy

def clone_graph_batch_in_chunks(batch, batch_size=1000):
    # 创建一个新的 Graph 对象
    cloned_batch = Graph()

    # 分批次克隆 node_feature
    num_nodes = batch.node_feature.size(0)
    new_node_feature = torch.zeros_like(batch.node_feature)
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        new_node_feature[start:end] = batch.node_feature[start:end].clone()
    cloned_batch.node_feature = new_node_feature

    # 分批次克隆 node_degree_existing
    new_node_degree_existing = torch.zeros_like(batch.node_degree_existing)
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        new_node_degree_existing[start:end] = batch.node_degree_existing[start:end].clone()
    cloned_batch.node_degree_existing = new_node_degree_existing

    # 克隆简单张量属性
    cloned_batch.edge_feature = batch.edge_feature.clone()
    cloned_batch.edge_index = batch.edge_index.clone()
    cloned_batch.edge_label = batch.edge_label.clone()
    cloned_batch.edge_label_index = batch.edge_label_index.clone()
    cloned_batch.edge_time = batch.edge_time.clone()
    cloned_batch.keep_ratio = batch.keep_ratio.clone()
    cloned_batch.node_degree_new = batch.node_degree_new.clone()
    cloned_batch.node_label_index = batch.node_label_index.clone()

    # 克隆简单列表或标量属性
    cloned_batch.directed = copy.deepcopy(batch.directed)
    cloned_batch.negative_label_val = copy.deepcopy(batch.negative_label_val)
    cloned_batch.node_cells = copy.deepcopy(batch.node_cells)
    cloned_batch.task = copy.deepcopy(batch.task)

    # 分批次克隆 list 类型的 node_states
    cloned_batch.node_states = []
    for state in batch.node_states:
        if isinstance(state, torch.Tensor):
            # 如果是Tensor，进行分批次克隆
            state_size = state.size(0)  # 假设第一维是节点数量
            new_state = torch.zeros_like(state)
            for start in range(0, state_size, batch_size):
                end = min(start + batch_size, state_size)
                new_state[start:end] = state[start:end].clone()
            cloned_batch.node_states.append(new_state)
        else:
            # 如果不是Tensor，直接深度复制
            cloned_batch.node_states.append(copy.deepcopy(state))
    return cloned_batch

class GNN(nn.Module):
    r"""The General GNN model"""

    def __init__(self, dim_in, dim_out, **kwargs):
        r"""Initializes the GNN model.

        Args:
            dim_in, dim_out: dimensions of in and out channels.
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNStage1= stage_dict['stack1']
        GNNHead = head_dict[cfg.dataset.task]
        # Currently only for OGB datasets
        # node_encoder在cfg中是false
        
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim

        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            EdgeEncoder1 = edge_encoder_dict['roland_general1']
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            self.edge_encoder1 = EdgeEncoder1(1)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)
        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner

        ##################静态层特征提取模块#################
        self.mp1=GNNStage1(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp1)
        
        if cfg.gnn.layers_mp >= 1:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            
            self.attention = nn.MultiheadAttention(embed_dim=d_in, num_heads=8)

            self.general = nn.Sequential(nn.Linear(64, 64),nn.ReLU())
            
            # self.ccn_module=CCN()
            # self.attention_module = AttentionModule(nb_nodes=38918,hid_units=64) # MT:116408
            
            self.attention_module = MultiHeadAttentionModule(nb_nodes=cfg.dataset.num_nodes,hid_units=64,num_heads=8)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)
        self.apply(init_weights)

    def forward(self, batch1, batch2, batch3):

        alpha = cfg.gnn.alpha_value

        batch1=self.edge_encoder1(batch1)
        batch1=self.edge_encoder_bn(batch1)
        batch1=self.pre_mp(batch1)

        batch2=self.edge_encoder1(batch2)
        batch2=self.edge_encoder_bn(batch2)
        batch2=self.pre_mp(batch2)

        batch3=self.edge_encoder1(batch3)
        batch3=self.edge_encoder_bn(batch3)
        batch3=self.pre_mp(batch3)

        copied_batch1 = clone_graph_batch_in_chunks(batch1, batch_size=1000)
        copied_batch2 = clone_graph_batch_in_chunks(batch2, batch_size=1000)
        copied_batch3 = clone_graph_batch_in_chunks(batch3, batch_size=1000)

        '''
        copied_batch1 = batch1.clone()
        copied_batch2 = batch2.clone()
        copied_batch3 = batch3.clone()'''
        
        # 捕捉层内嵌入（within layer）
        batch1=self.mp(copied_batch1)
        batch2=self.mp(copied_batch2)
        batch3=self.mp(copied_batch3)

        # 捕捉层间嵌入（across layer）
        # 初始化结果张量
        if cfg.dataset.name == 'cannes.txt':
            num_nodes = 438537
        across_feature = torch.zeros((num_nodes, 64), device=cfg.device)
        batch_size = 1000
        
        # 分批次操作
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)

            # 提取当前批次的节点特征
            sub_batch1 = copied_batch1.node_feature[start:end].unsqueeze(0)
            sub_batch2 = copied_batch2.node_feature[start:end].unsqueeze(0)
            sub_batch3 = copied_batch3.node_feature[start:end].unsqueeze(0)

            # 调用注意力函数
            sub_feature = self.attention(sub_batch1, sub_batch2, sub_batch3)  # 输出形状: [1, batch_size, feature_dim]
            sub_feature = sub_feature[0].squeeze(0)  # 去掉多余的维度，形状变为 [batch_size, feature_dim]

            # 对当前批次进行更新
            copied_batch1.node_feature[start:end] = (
                alpha * sub_feature + (1 - alpha) * copied_batch1.node_feature[start:end]
            )
            copied_batch2.node_feature[start:end] = (
                alpha * sub_feature + (1 - alpha) * copied_batch2.node_feature[start:end]
            )
            copied_batch3.node_feature[start:end] = (
                alpha * sub_feature + (1 - alpha) * copied_batch3.node_feature[start:end]
            )

            # 调用 general 函数对当前批次的特征进行处理
            copied_batch1.node_feature[start:end] = self.general(copied_batch1.node_feature[start:end])
            copied_batch2.node_feature[start:end] = self.general(copied_batch2.node_feature[start:end])
            copied_batch3.node_feature[start:end] = self.general(copied_batch3.node_feature[start:end])

        node_states = list()
        node_states = [batch1.node_states, batch2.node_states, batch3.node_states]

        '''
        pred1,label1 = self.post_mp(batch1)
        pred2,label2 = self.post_mp(batch2)
        pred3,label3 = self.post_mp(batch3)

        preds = [pred1, pred2, pred3]
        labels = [label1, label2, label3]'''

        batch1 = self.post_mp(batch1)
        batch2 = self.post_mp(batch2)
        batch3 = self.post_mp(batch3)

        return batch1, batch2, batch3, node_states

register_network('gnn_recurrent_multi', GNN)