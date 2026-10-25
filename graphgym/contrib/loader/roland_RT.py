import os
from typing import List, Union

import re
import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler

from graphgym.config import cfg
from graphgym.contrib.loader.loader_utils import make_graph_snapshot
from graphgym.register import register_loader
from graphgym.utils.stats import node_degree

def filter_friendship():
    node_dict = {}
    with open("/root/autodl-tmp/demo/RT_friend/node_dict.txt", "r") as dict_file:
        for line in dict_file:
            node_id, node_name = line.strip().split()
            node_dict[node_name] = node_id

    filtered_lines = []
    with open("/root/autodl-tmp/demo/roland_public_data/Friendship.txt", "r") as friendship_file:
        for line in friendship_file:
            nodes = line.strip().split()
            if all(node in node_dict for node in nodes):
                filtered_nodes = [node_dict[node] for node in nodes]
                filtered_lines.append(" ".join(filtered_nodes) + "\n")

    with open("/root/autodl-tmp/demo/RT_friend/sub_friend.txt", "w") as output_file:
        output_file.writelines(filtered_lines)
    
def friend_add_time(snapshot_list,g_all_1):
    # Initialize an empty list to store the time-sliced sub_friend graphs.
    friend_time_list = []

    # Load the sub_friend graph from the file.
    with open('/root/autodl-tmp/demo/RT_friend/sub_friend.txt', 'r') as file:
        sub_friend_content = file.readlines()

    # Process each snapshot and create the corresponding time-sliced sub_friend graph.
    for idx, snapshot in enumerate(snapshot_list):
        # Create a new Graph object for the time-sliced sub_friend graph.
        sub_friend_graph = Graph(
            node_feature=g_all_1.node_feature,
            edge_feature=g_all_1.edge_feature,
            edge_index=snapshot.edge_index,
            directed=True  # Assuming sub_friend.txt is a directed graph
        )

        # Extract the nodes in the current snapshot.
        snapshot_nodes = set(snapshot.edge_index.view(-1).tolist())

        # Filter out the edges in the sub_friend graph that belong to the current snapshot.
        sub_friend_edges = [line.strip().split() for line in sub_friend_content if all(int(edge) in snapshot_nodes for edge in line.strip().split()[:2])]

        # Convert sub_friend_edges to edge_index and add it to the sub_friend_graph.
        sub_friend_edges = [[int(edge[0]), int(edge[1])] for edge in sub_friend_edges]
        sub_friend_edges = torch.tensor(sub_friend_edges, dtype=torch.long).t()
        sub_friend_graph.edge_index = sub_friend_edges

        # Append the time-sliced sub_friend graph to the friend_time_list.
        friend_time_list.append(sub_friend_graph)

        # Save the time-sliced sub_friend graph to a file with the format "sub_friend_{idx}.txt".
        sub_friend_file_path = f'/root/autodl-tmp/demo/RT_friend/sub_friend_{idx}.txt'
        with open(sub_friend_file_path, 'w') as sub_friend_file:
            for edge in sub_friend_edges.t().tolist():
                sub_friend_file.write(f"{edge[0]} {edge[1]}\n")

    return friend_time_list

def load_single_dataset(dataset_dir: str) -> Graph:
    # Load dataset using dask for fast parallel loading.
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST', 'TIMESTAMP']
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)

    # Node IDs of this dataset start from 1.
    df_trans['SRC'] -= 1
    df_trans['DST'] -= 1

    num_nodes = len(set(df_trans['SRC']) | set(df_trans['DST']))
    num_edges = len(set(zip(df_trans['SRC'], df_trans['DST'])))

    # 创建包含所有节点的集合
    all_nodes = set(df_trans['SRC']).union(set(df_trans['DST']))

    # 创建节点ID到整数键值的映射字典
    node_dict = {node: i for i, node in enumerate(sorted(all_nodes))}

    '''output_dir = '/root/autodl-tmp/demo/RT_friend/'
    output_file = output_dir + 'node_dict.txt'

    with open(output_file, 'w') as file:
        for key, value in node_dict.items():
            file.write(f"{key} {value}\n")
    
    filter_friendship()'''
    
    # Replace node IDs in edge_index with node_dict keys
    # df_trans['SRC'] = df_trans['SRC'].replace(node_dict).astype(int)
    # df_trans['DST'] = df_trans['DST'].replace(node_dict).astype(int)
    df_trans['SRC'] = df_trans['SRC'].map(node_dict)
    df_trans['DST'] = df_trans['DST'].map(node_dict)

    # print('num of nodes:', np.max(df_trans[['SRC', 'DST']].values) + 1)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIMESTAMP'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['TimestampScaled']].values).view(-1, 1)
    edge_index = torch.Tensor(
        df_trans[['SRC', 'DST']].values.transpose()).long()  # (2, E)
    
    # num_nodes = torch.max(edge_index) + 1

    # node_feature = node_degree(edge_index, n=num_nodes, mode='both')
    # node_feature = node_feature.view(num_nodes, 1)
    # use node degree as feature leads to info-leak risk.
    node_feature = torch.ones(num_nodes, 1)

    edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph

#---------单独处理静态层的函数--------------#
def load_static_dataset(dataset_dir: str) -> Graph:
    # Load dataset using dask for fast parallel loading.
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST']
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)

    # num_nodes = len(set(df_trans['SRC']) | set(df_trans['DST']))
    num_edges = len(set(zip(df_trans['SRC'], df_trans['DST'])))
    num_nodes=	256491

    # print('num of nodes:', np.max(df_trans[['SRC', 'DST']].values) + 1)
    edge_feature = torch.ones((num_edges, 1))
    edge_index = torch.Tensor(
        df_trans[['SRC', 'DST']].values.transpose()).long()  # (2, E)
    
    # num_nodes = torch.max(edge_index) + 1

    # node_feature = node_degree(edge_index, n=num_nodes, mode='both')
    # node_feature = node_feature.view(num_nodes, 1)
    # use node degree as feature leads to info-leak risk.
    node_feature = torch.ones(num_nodes, 1)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        directed=True
    )

    return graph

def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
# tensor([ True,  True,  True,  ..., False, False, False])

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
    dir="/root/autodl-tmp/demo/RT_friend/sub_friend.txt"
    g_all_1= load_static_dataset(dir)

    if not snapshot:
        return g_all
    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)

    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq,
                                            is_hetero=False)
        
    # friend_list=friend_add_time(snapshot_list,g_all_1)
    num_nodes = len(g_all_1.node_feature)

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    return snapshot_list

def load_generic_dataset(format, name, dataset_dir):
    if format == 'RT':
        graphs = load_generic(os.path.join(dataset_dir, name),
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)
        if cfg.dataset.split_method == 'chronological_temporal':
            # return graphs
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 2:
                    filtered_graphs.append(g)
            return filtered_graphs
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 10:
                    filtered_graphs.append(g)
            return filtered_graphs

register_loader('roland_RT', load_generic_dataset)