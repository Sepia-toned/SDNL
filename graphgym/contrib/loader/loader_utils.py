"""
Helper functions and utilities for data loaders, designed and made for datasets
used in ROLAND project.

Mar. 31, 2021.
"""
import numpy as np
import pandas as pd
from deepsnap.graph import Graph
from typing import List
import torch
from graphgym.config import cfg


def make_graph_snapshot1(g_all: Graph,
                        snapshot_freq: str,
                        is_hetero: bool = True) -> List[Graph]:
    """
    Constructs a list of graph snapshots based from g_all using g_all.edge_time
    and provided snapshot_freq.

    Args:
        g_all: the entire graph object, g_all must have a edge_time attribute,
            g_all.edge_time consists of unix timestamp of edge time.
        snapshot_freq: snapshot frequency, must be one of
            'D': daily, 'W': weekly, and 'M': monthly.
        is_hetero: whether the graph is heterogeneous.

    Return:
        A list of graph object, each graph snapshot has edge level information
            (edge_feature, edge_time, etc) of only edges in that time period.
            However, every graph snapshot has the same and full node level
            information (node_feature, node_type, etc).
    """
    # Arg check.
    if not hasattr(g_all, 'edge_time'):
        raise KeyError('Temporal graph needs to have edge_time attribute.')

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        raise ValueError(f'Unsupported snapshot freq: {snapshot_freq}.')

    snapshot_freq = snapshot_freq.upper()
    t = g_all.edge_time.numpy().astype(np.int64)  # all timestamps.

    period_split = pd.DataFrame(
        {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time))
    )

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'}  # month of year.

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    # e.g., dictionary w/ key = (2021, 3) and val = array(edge IDs).

    periods = sorted(list(period2id.keys()))  # ascending order.
    # 获取后一半的元素
    periods_dy = periods[len(periods)//2:]
    periods_st = periods[:len(periods)//2]

    # alternatively, sorted(..., key=lambda x: x[0] + x[1]/1000).
    snapshot_list = list()
    for p in periods_dy:
        # unique IDs of edges in this period.
        period_members = period2id[p]

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed,
            list_n_type=g_all.list_n_type if is_hetero else None,
            list_e_type=g_all.list_e_type if is_hetero else None,
        )
        if is_hetero and hasattr(g_all, 'node_type'):
            g_incr.node_type = g_all.node_type
            g_incr.edge_type = g_all.edge_type[period_members]
        snapshot_list.append(g_incr)
    
    # 保存为静态网络
    edges_st = []
    for p in periods_st:
        period_members = period2id[p]
        edge_index=g_all.edge_index[:, period_members]
        edges_st.append(edge_index)

    # 将 edges_st 中的所有边拼接成一个张量
    all_edges = torch.cat(edges_st, dim=1)  # 拼接得到 2 x M 的张量，其中 M 是所有边的总数

    # 去重操作
    unique_edges = torch.unique(all_edges, dim=1)  # 按列去重，结果仍是 2 x K 的张量

    if cfg.dataset.name == 'CollegeMsg.txt':
        output_file = '/home/ly/SDNL/CollegeMsg_T/sub_friend.txt'
    elif cfg.dataset.name == 'bitcoinalpha.csv':
        output_file = '/home/ly/SDNL/Bitcoinalpha_T/sub_friend.txt'
    elif cfg.dataset.name == 'bitcoinotc.csv':
        output_file = '/home/ly/SDNL/Bitcoinotc_T/sub_friend.txt'
    with open(output_file, 'w') as file:
        for edge in unique_edges.t():  # 转置为每列一行
            file.write(f"{edge[0].item()} {edge[1].item()}\n")  # 写入文件，两个节点用空格分隔

    return snapshot_list

def make_graph_snapshot(g_all: Graph,
                        snapshot_freq: str,
                        is_hetero: bool = True) -> List[Graph]:
    """
    Constructs a list of graph snapshots based from g_all using g_all.edge_time
    and provided snapshot_freq.

    Args:
        g_all: the entire graph object, g_all must have a edge_time attribute,
            g_all.edge_time consists of unix timestamp of edge time.
        snapshot_freq: snapshot frequency, must be one of
            'D': daily, 'W': weekly, and 'M': monthly.
        is_hetero: whether the graph is heterogeneous.
 
    Return:
        A list of graph object, each graph snapshot has edge level information
            (edge_feature, edge_time, etc) of only edges in that time period.
            However, every graph snapshot has the same and full node level
            information (node_feature, node_type, etc).
    """
    # Arg check.
    if not hasattr(g_all, 'edge_time'):
        raise KeyError('Temporal graph needs to have edge_time attribute.')

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        raise ValueError(f'Unsupported snapshot freq: {snapshot_freq}.')

    snapshot_freq = snapshot_freq.upper()
    t = g_all.edge_time.numpy().astype(np.int64)  # all timestamps.

    period_split = pd.DataFrame(
        {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time))
    )

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'}  # month of year.

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    # e.g., dictionary w/ key = (2021, 3) and val = array(edge IDs).

    periods = sorted(list(period2id.keys()))  # ascending order.

    # alternatively, sorted(..., key=lambda x: x[0] + x[1]/1000).
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed,
            list_n_type=g_all.list_n_type if is_hetero else None,
            list_e_type=g_all.list_e_type if is_hetero else None,
        )
        if is_hetero and hasattr(g_all, 'node_type'):
            g_incr.node_type = g_all.node_type
            g_incr.edge_type = g_all.edge_type[period_members]
        snapshot_list.append(g_incr)
    return snapshot_list
