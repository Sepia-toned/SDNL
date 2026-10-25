"""
Loader for the CollegeMsg temporal network.

For more information: https://snap.stanford.edu/data/CollegeMsg.html

Mar. 31, 2021
"""
import os
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler

from graphgym.config import cfg
from graphgym.contrib.loader.loader_utils import make_graph_snapshot, make_graph_snapshot1
from graphgym.register import register_loader
from graphgym.utils.stats import node_degree

'''def load_single_dataset(dataset_dir: str) -> Graph:
    # Load dataset using dask for fast parallel loading.
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST', 'TIMESTAMP','LAYER']
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)

    print('边数目:', len(df_trans))
    print('节点数目:', np.max(df_trans[['SRC', 'DST']].values) + 1)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIMESTAMP'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['TimestampScaled']].values).view(-1, 1)
    edge_index = torch.Tensor(
        df_trans[['SRC', 'DST']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    # node_feature = node_degree(edge_index, n=num_nodes, mode='both')
    # node_feature = node_feature.view(num_nodes, 1)
    # use node degree as feature leads to info-leak risk.
    node_feature = torch.ones(num_nodes, 1)

    print('feature_node_int_num: ', node_feature.max() + 1)

    edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph'''

def load_single_dataset(dataset_dir: str) -> tuple:
    # Load dataset using pandas
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST', 'TIMESTAMP', 'LAYER']
    assert not np.any(pd.isna(df_trans).values)  # Ensure no missing data
    df_trans.reset_index(drop=True, inplace=True)

    print('边数目:', len(df_trans))
    print('节点数目:', np.max(df_trans[['SRC', 'DST']].values) + 1)

    # Scale the timestamps to a range between 0 and 2
    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIMESTAMP'].values.reshape(-1, 1))

    # Prepare edge features
    edge_feature = torch.Tensor(df_trans[['TimestampScaled']].values).view(-1, 1)

    # Prepare edge_index (SRC, DST)
    edge_index = torch.Tensor(df_trans[['SRC', 'DST']].values.transpose()).long()

    # Get the number of nodes
    num_nodes = torch.max(edge_index) + 1

    # Assign node features (here using a simple constant feature for all nodes)
    node_feature = torch.ones(num_nodes, 1)

    print('feature_node_int_num: ', node_feature.max() + 1)

    # Prepare edge times
    edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

    # Split the dataframe by LAYER to create 3 separate graphs
    graph1_df = df_trans[df_trans['LAYER'] == 'RE']
    graph2_df = df_trans[df_trans['LAYER'] == 'MT']
    graph3_df = df_trans[df_trans['LAYER'] == 'RT']

    # Create Graph 1
    edge_feature1 = torch.Tensor(graph1_df[['TimestampScaled']].values).view(-1, 1)
    edge_index1 = torch.Tensor(graph1_df[['SRC', 'DST']].values.transpose()).long()
    edge_time1 = torch.FloatTensor(graph1_df['TIMESTAMP'].values)

    graph1 = Graph(
        node_feature=node_feature,  # Node features (same for all graphs)
        edge_index=edge_index1, 
        edge_feature=edge_feature1,
        edge_time=edge_time1,
        directed=True
    )

    # Create Graph 2
    edge_feature2 = torch.Tensor(graph2_df[['TimestampScaled']].values).view(-1, 1)
    edge_index2 = torch.Tensor(graph2_df[['SRC', 'DST']].values.transpose()).long()
    edge_time2 = torch.FloatTensor(graph2_df['TIMESTAMP'].values)

    graph2 = Graph(
        node_feature=node_feature,  # Node features (same for all graphs)
        edge_index=edge_index2, 
        edge_feature=edge_feature2,
        edge_time=edge_time2,
        directed=True
    )

    # Create Graph 3
    edge_feature3 = torch.Tensor(graph3_df[['TimestampScaled']].values).view(-1, 1)
    edge_index3 = torch.Tensor(graph3_df[['SRC', 'DST']].values.transpose()).long()
    edge_time3 = torch.FloatTensor(graph3_df['TIMESTAMP'].values)
    graph3 = Graph(
        node_feature=node_feature,  # Node features (same for all graphs)
        edge_index=edge_index3, 
        edge_feature=edge_feature3,
        edge_time=edge_time3,
        directed=True
    )
    return graph1, graph2, graph3

def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
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
    g_all_1, g_all_2, g_all_3 = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all
    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list_1 = split_by_seconds(g_all_1, freq)
        snapshot_list_2 = split_by_seconds(g_all_2, freq)
        snapshot_list_3 = split_by_seconds(g_all_3, freq)

    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq,
                                            is_hetero=False)

    '''num_nodes1 = g_all_1.edge_index.max() + 1
    num_nodes2 = g_all_2.edge_index.max() + 1
    num_nodes3 = g_all_3.edge_index.max() + 1'''

    if cfg.dataset.name == 'cannes.txt':
        num_nodes = 438537
    
    elif cfg.dataset.name == 'mlking.txt':
        num_nodes = 327707

    for g_snapshot in snapshot_list_1:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    for g_snapshot in snapshot_list_2:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    for g_snapshot in snapshot_list_3:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    return snapshot_list_1, snapshot_list_2, snapshot_list_3

def load_generic_dataset(format, name, dataset_dir):
    if format == 'multi':
        graphs1, graphs2, graphs3 = load_generic(os.path.join(dataset_dir, name),
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
            filtered_graphs_1 = list()
            filtered_graphs_2 = list()
            filtered_graphs_3 = list()
            for g in graphs1:
                if g.num_edges >= 10:
                    filtered_graphs_1.append(g)
            for g in graphs2:
                if g.num_edges >= 10:
                    filtered_graphs_2.append(g)
            for g in graphs3:
                if g.num_edges >= 10:
                    filtered_graphs_3.append(g)
                             
            return filtered_graphs_1, filtered_graphs_2, filtered_graphs_3

register_loader('multi_networks', load_generic_dataset)