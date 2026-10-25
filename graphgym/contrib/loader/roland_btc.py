"""
Data loader for bitcoin datasets.
Mar. 27, 2021
"""
import os
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from graphgym.config import cfg
from graphgym.register import register_loader


def load_single_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    df_trans['TIME'] = df_trans['TIME'].astype(np.int).astype(np.float)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    print('节点数量:',num_nodes)
    node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
        edge_feature = torch.cat((edge_feature, edge_feature.clone()), dim=0)
        reversed_idx = torch.stack([edge_index[1], edge_index[0]]).clone()
        edge_index = torch.cat((edge_index, reversed_idx), dim=1)
        edge_time = torch.cat((edge_time, edge_time.clone()))

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph

def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()

    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

    return snapshot_list

def make_graph_snapshot1(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()
    
    # 构造时间戳和时间列
    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))
    # 修改freq_map，增加对2个月的支持
    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m',  # month of year.
                '12M': lambda x: (x.month - 1) // 10 + 1  # 每2个月划分一个组
                }

    # 提取年份和按所选粒度（如2个月）分组
    period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

    # 根据所选的snapshot_freq来进行分组（例如：'2M'）
    if snapshot_freq == '12M':
        period_split['SubYearFlag'] = period_split['TransactionTime'].apply(freq_map['12M']).astype(int)
    else:
        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[snapshot_freq]).astype(int)

    # 按年和时间段分组
    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    periods_dy = periods[len(periods)//2:]
    periods_st = periods[:len(periods)//2]
    snapshot_list = list()

    for p in periods_dy:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

    # 保存为静态网络
    edges_st = []
    for p in periods_st:
        period_members = period2id[p]
        edge_index=g_all.edge_index[:, period_members]
        edge_feature = g_all.edge_feature[period_members, :]

        new_row = edge_feature[:, 0]  # (N,)
        extended_edge_index = torch.cat([edge_index, new_row.unsqueeze(0)], dim=0)

        edges_st.append(extended_edge_index)

    # 将 edges_st 中的所有边拼接成一个张量
    all_edges = torch.cat(edges_st, dim=1)

    #--------------------------------- 去重操作 ----------------------------------#

    # 假设 all_edges 已经定义好
    edge_index = all_edges[:2, :]  # (2, M)，边的源节点和目标节点
    edge_weights = all_edges[2, :]  # (M,)，边的权重

    # 确保 edge_index 和 edge_weights 是 PyTorch 张量
    edge_index = torch.tensor(edge_index)
    edge_weights = torch.tensor(edge_weights)

    # 获取边的源节点和目标节点
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # 组合边的源节点和目标节点，形成边的唯一标识
    edge_pairs = torch.stack([source_nodes, target_nodes], dim=1)

    # 使用 unique 查找唯一的边对
    unique_pairs, inverse_indices = edge_pairs.unique(dim=0, return_inverse=True)

    # 统计每个边对出现的次数
    counts = torch.zeros(unique_pairs.shape[0], dtype=torch.int)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(edge_weights, dtype=torch.int))

    # 找到重复的边对：counts > 1
    duplicate_indices = counts > 1

    # 打印重复的边对及其出现次数
    if duplicate_indices.sum() > 0:
        print(f"Found {duplicate_indices.sum().item()} duplicate edges:")
        for idx in duplicate_indices.nonzero(as_tuple=True)[0]:
            print(f"Edge: {unique_pairs[idx].tolist()} appears {counts[idx].item()} times")
    else:
        print("No duplicate edges found.")

    # 计算总边权重，聚合重复边对的权重
    total_weights = torch.zeros(unique_pairs.shape[0], dtype=edge_weights.dtype)
    total_weights.scatter_add_(0, inverse_indices, edge_weights)

    # 计算每个唯一边对的平均权重
    average_weights = total_weights / counts.float()

    # 合并成三行的最终结果
    final_edge_index = torch.cat([unique_pairs.T, average_weights.unsqueeze(0)], dim=0)

    # final_edge_index 的形状是 (3, N)，N 是唯一边对的数量
    print(final_edge_index.size())

    if cfg.dataset.name == 'bitcoinalpha.csv':
        output_file = '/home/ly/SDNL/Bitcoinalpha_T/sub_friend.txt'
    elif cfg.dataset.name == 'bitcoinotc.csv':
        output_file = '/home/ly/SDNL/Bitcoinotc_T/sub_friend.txt'
    with open(output_file, 'w') as file:
        for edge in final_edge_index.t():  # 转置为每列一行
            file.write(f"{int(edge[0].item())} {int(edge[1].item())} {edge[2].item()} \n") 
    
    return snapshot_list

def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list

def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M','12M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)
    else:
        snapshot_list = make_graph_snapshot1(g_all, snapshot_freq)
    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    # check snapshots ordering.
    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    return snapshot_list

def load_btc_dataset(format, name, dataset_dir):
    # Keep this experimental loader isolated from the default bitcoin loader.
    if format == 'bitcoin':
        graphs = load_generic(os.path.join(dataset_dir, name),
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)
        if cfg.dataset.split_method == 'chronological_temporal':
            return graphs
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 100:
                    filtered_graphs.append(g)
            return filtered_graphs

register_loader('roland_btc', load_btc_dataset)