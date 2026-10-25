import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
        
        self.bn = nn.BatchNorm1d(hid_units)
        self.prelu = nn.PReLU()
        
    def forward(self, batch, batch1):
        temp_list=[]
        nodes_ref = None
        for idx in range(self.num_heads):
            temp,nodes= self.heads[idx](batch,batch1)
            temp_list.append(temp)
            if nodes_ref is None:
                nodes_ref = nodes

        temp1=torch.mean(torch.cat(temp_list, 0), 0)
        
        '''temp1=F.normalize(temp1, p=2, dim=-1)
        temp1=self.prelu(temp1)'''
        
        # Keep full node_feature shape for downstream heads; only update
        # nodes participating in this fusion step.
        fused_feature = batch.node_feature.clone()
        fused_feature[nodes_ref] = temp1
        batch.node_feature = fused_feature
        # batch.node_feature = 0.5*(batch.node_feature+batch1.node_feature)

        batch.node_feature=self.bn(batch.node_feature)
        batch.node_feature=self.prelu(batch.node_feature)

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
        nodes = torch.unique(batch.edge_index.view(-1), sorted=True)
        num_nodes = nodes.size(0)
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
            
            # self.ccn_module=CCN()
            # self.attention_module = AttentionModule(nb_nodes=38918,hid_units=64) # MT:116408
            
            self.attention_module = MultiHeadAttentionModule(nb_nodes=cfg.dataset.num_nodes,hid_units=64,num_heads=8)
            d_in = self.mp.dim_out
            # Eq.22: node-level general refinement after static-dynamic fusion.
            self.fusion_general = GeneralMultiLayer('linear', 1, d_in, d_in,
                                                    dim_inner=d_in,
                                                    final_act=True)
            
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)
        self.apply(init_weights)

    def forward(self, batch,batch1):

        batch1=self.edge_encoder1(batch1)
        # BatchNorm1d requires at least 2 samples per channel in training.
        # Some snapshots may contain only one static edge feature row.
        if (not hasattr(batch1, 'edge_feature')) or batch1.edge_feature.size(0) > 1:
            batch1=self.edge_encoder_bn(batch1)
        batch1=self.pre_mp(batch1)
        batch1=self.mp1(batch1)
        
        '''for module in self.children():
            # if module==self.post_mp :
            if module==self.attention_module:
                break
            batch1 = module(batch1)'''

        for module in self.children():
            if module==self.attention_module:
                batch = module(batch,batch1)
                continue
            elif module==self.mp1:
                continue
            elif module==self.edge_encoder1:
                continue
            batch=module(batch)     
        return batch

register_network('gnn_recurrent_att', GNN)